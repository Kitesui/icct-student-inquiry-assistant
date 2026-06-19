// ============================================================================
//  Casper AI Backend Server — ICCT Colleges AI Chatbot (Capstone Project)
// ============================================================================
//  Tech stack : Express  ·  @google/genai (next-gen SDK)  ·  ES Modules
//  Responsibilities:
//    1. Load & parse 3 CSV knowledge-base files at startup.
//    2. Perform keyword-based RAG retrieval on every chat request.
//    3. Forward the retrieved context + conversation history to Gemini.
//    4. Intercept the FALLBACK_TRIGGER sentinel and return a ticket prompt.
// ============================================================================

import dotenv from "dotenv";
import express from "express";
import cors from "cors";
import fs from "fs";
import { GoogleGenAI } from "@google/genai";
import Groq from "groq-sdk";
import { createClient } from "@supabase/supabase-js";
import { createRequire } from "module";
import bcrypt from "bcrypt";
import { fileURLToPath } from "url";

const require = createRequire(import.meta.url);
const path = require('path');
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load .env relative to the server script location (root folder)
require('dotenv').config({ path: require('path').join(__dirname, '../.env') });

// ── Constants ───────────────────────────────────────────────────────────────
const PORT = 5000;
const MODEL_NAME = "gemini-2.0-flash-lite";

// Paths to the 3 knowledge-base CSV files (relative to data directory)
const CSV_FILES = [
  path.join(__dirname, "../data/GENERAL ACADEMIC POLICIES.csv"),
  path.join(__dirname, "../data/ADMISSION.csv"),
  path.join(__dirname, "../data/COURSE OFFERING.csv"),
];

// ── System Instruction Builder for Gemini ───────────────────────────────────
function getSystemInstruction(language) {
  return `You are the official ICCT Colleges Student Support Assistant. Your job is to answer student inquiries in a clear, scannable, and well-structured conversational format using the provided university handbook context segments.

CRITICAL HANDLING RULES:
1. Prioritize the provided context blocks as your absolute source of truth for procedures, policies, names, and contact details.
2. If a student mentions specific waiting timelines (like '3 days', 'isang linggo') or personal emotional panic that isn't explicitly written in the dry handbook rules, do not panic or return an empty string. Use your natural AI reasoning to calmly map their situation to the closest corresponding handbook procedure (e.g., explaining the standard verification timeline, checking the student ledger, or advising them on who to contact).
3. If the provided context block completely lacks any relevant topics or procedures matching the user's operational problem, output exactly 'FALLBACK_TRIGGER' so our system can safely offer an official administrative support ticket.

CRITICAL FORMATTING RULES:
1. USE LINE BREAKS AND PARAGRAPHS: Never output large walls of dense text. Break down your thoughts into short paragraphs (2-3 sentences max).
2. USE BULLET POINTS OR NUMBERED LISTS: When listing requirements, steps, or conditions (like when an SOG is needed), always format them as a clean vertical list using asterisks (*) or numbers (1., 2.). Ensure there is a line break before and after lists.
3. USE BOLDING SPARINGLY: Use **bold text** only for critical terms, document names (e.g., **Summary of Grades**), or important reminders.
4. TONE AND LANGUAGE: Maintain an empathetic, helpful tone. You MUST generate the final response completely and strictly in ${language}.

CRITICAL LANGUAGE ENFORCEMENT: Respond strictly in ${language}. If the handbook context is in English, translate it to ${language} when generating your reply.`;
}

// ── Google GenAI Initialisation ─────────────────────────────────────────────
// The SDK reads the API key we pass here; store it in .env as GEMINI_API_KEY.
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

// ── Supabase Cloud Database Initialisation ──────────────────────────────────
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

// ── Helper: embedWithRetry to handle rate limits ────────────────────────────
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function embedWithRetry(content, maxRetries = 4) {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const result = await ai.models.embedContent({
        model: "gemini-embedding-001",
        contents: content,
        config: { outputDimensionality: 1536 },
      });
      return result.embeddings[0].values;
    } catch (err) {
      const is429 = err.message?.includes("429") || err.message?.includes("RESOURCE_EXHAUSTED");
      if (is429 && attempt < maxRetries) {
        let backoff = Math.pow(2, attempt) * 1000;
        const match = err.message?.match(/Please retry in (\d+(?:\.\d+)?)s/i);
        if (match) {
          const seconds = parseFloat(match[1]);
          backoff = Math.ceil(seconds * 1000) + 1000;
        }

        // If the backoff is too long (more than 5 seconds), abort and throw the error to prevent gateway timeout
        if (backoff > 5000) {
          console.warn(`  ⏳ Rate limit backoff is too long (${backoff / 1000}s) — aborting retry.`);
          throw err;
        }

        console.warn(`  ⏳ Rate limited on embedding API — sleeping for ${backoff / 1000}s (attempt ${attempt + 1}/${maxRetries})…`);
        await sleep(backoff);
      } else {
        throw err;
      }
    }
  }
}

async function generateContentWithRetry(model, config, contents, maxRetries = 4) {
  // ── Convert Gemini-style contents array to Groq/OpenAI messages format ──
  const messages = [];

  // Add system instruction if provided
  if (config?.systemInstruction) {
    messages.push({ role: "system", content: config.systemInstruction });
  }

  // Convert Gemini turns { role, parts: [{ text }] } → { role, content }
  for (const turn of contents) {
    const role = turn.role === "model" ? "assistant" : "user";
    const content = turn.parts?.map(p => p.text).join("") ?? "";
    messages.push({ role, content });
  }

  // Use Groq model name (ignore the Gemini model name passed in)
  const groqModel = "llama-3.3-70b-versatile";

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const completion = await groq.chat.completions.create({
        model: groqModel,
        messages,
        temperature: 0.7,
        max_tokens: 1024,
      });

      const text = completion.choices[0]?.message?.content ?? "";
      // Return a response object that mirrors the Gemini response shape
      return { text };
    } catch (err) {
      const is429 = err.status === 429 || err.message?.includes("429") || err.message?.includes("rate_limit");
      if (is429 && attempt < maxRetries) {
        let backoff = Math.pow(2, attempt) * 1000;
        const match = err.message?.match(/try again in ([\d.]+)s/i);
        if (match) {
          backoff = Math.ceil(parseFloat(match[1]) * 1000) + 500;
        }
        if (backoff > 5000) {
          console.warn(`  ⏳ Rate limit backoff is too long (${backoff / 1000}s) — aborting retry.`);
          throw err;
        }
        console.warn(`  ⏳ Rate limited on Groq API — sleeping for ${backoff / 1000}s (attempt ${attempt + 1}/${maxRetries})…`);
        await sleep(backoff);
      } else {
        throw err;
      }
    }
  }
}

// ── In-Memory Knowledge Base ────────────────────────────────────────────────
// Every row across all 3 CSVs becomes an object { topic, context, keywords }
// stored in this array.  It is populated once at server boot via loadCSVFiles().
let globalKnowledgeBase = [];

// ============================================================================
//  CSV PARSING HELPERS
// ============================================================================

/**
 * parseCSVLine — Splits a single CSV line into an array of field values.
 *
 * Because the CSV files contain multi-line quoted fields (e.g. long paragraphs
 * with commas), we use a simple state-machine parser instead of a naïve
 * `.split(",")`.  It toggles an "inside quotes" flag whenever it encounters
 * a double-quote character and only splits on commas that appear *outside*
 * quoted regions.
 *
 * @param {string} line – One complete logical CSV record.
 * @returns {string[]}  – Array of field values (quotes stripped).
 */
function parseCSVLine(line) {
  const fields = [];
  let current = "";
  let insideQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];

    if (char === '"') {
      // Handle escaped quotes ("") inside a quoted field
      if (insideQuotes && i + 1 < line.length && line[i + 1] === '"') {
        current += '"';
        i++; // skip the second quote
      } else {
        insideQuotes = !insideQuotes;
      }
    } else if (char === "," && !insideQuotes) {
      fields.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }

  // Push the final field
  fields.push(current.trim());
  return fields;
}

/**
 * parseCSV — Converts raw CSV text into an array of knowledge-base entries.
 *
 * Strategy:
 *   1. The CSV files have a header row (row 3 in the file) with columns
 *      "TOPIC/INTENT", "FACTUAL CONTEXT(THE ANSWER)", "KEYWORDS/TAGS"
 *      located at column indices 2, 3, and 4 (0-based).
 *   2. Many rows are blank separators — we skip any row where all three
 *      target columns are empty.
 *   3. Because some "Context" cells span multiple lines (they are wrapped in
 *      double-quotes), we first join broken lines back together before
 *      splitting into logical records.
 *
 * @param {string} rawText – Full file contents of one CSV.
 * @returns {{ topic: string, context: string, keywords: string }[]}
 */
function parseCSV(rawText) {
  const entries = [];

  // ── Step 1: Reassemble multi-line quoted fields ───────────────────────
  // Walk through raw lines and merge any that sit inside an unclosed quote.
  const rawLines = rawText.split("\n");
  const logicalLines = [];
  let buffer = "";
  let openQuotes = false;

  for (const raw of rawLines) {
    const line = raw.replace(/\r$/, ""); // strip trailing CR

    if (!openQuotes) {
      buffer = line;
    } else {
      buffer += "\n" + line; // continue accumulating
    }

    // Count unescaped quotes to decide if the field is still open
    const quoteCount = (buffer.match(/"/g) || []).length;
    openQuotes = quoteCount % 2 !== 0;

    if (!openQuotes) {
      logicalLines.push(buffer);
      buffer = "";
    }
  }
  // Flush anything remaining (shouldn't happen with well-formed CSVs)
  if (buffer) logicalLines.push(buffer);

  // ── Step 2: Parse each logical line into fields ───────────────────────
  for (const line of logicalLines) {
    const fields = parseCSVLine(line);

    // Columns of interest are at indices 2, 3, 4
    const topic   = (fields[2] || "").trim();
    const context = (fields[3] || "").trim();
    const keywords = (fields[4] || "").trim();

    // Skip blank rows and the header row itself
    if (!topic || topic === "TOPIC/INTENT") continue;

    entries.push({ topic, context, keywords });
  }

  return entries;
}

/**
 * loadCSVFiles — Reads all 3 CSV data files from disk, parses them, and
 * merges the results into `globalKnowledgeBase`.
 *
 * Called once at server startup.  If a file is missing, it logs a warning
 * and continues with the remaining files so the server doesn't crash during
 * development if one CSV is temporarily absent.
 */
function loadCSVFiles() {
  globalKnowledgeBase = [];

  for (const filePath of CSV_FILES) {
    const resolved = path.resolve(filePath);
    try {
      const raw = fs.readFileSync(resolved, "utf-8");
      const entries = parseCSV(raw);
      globalKnowledgeBase.push(...entries);
      console.log(`  ✓ Loaded ${entries.length} entries from ${path.basename(resolved)}`);
    } catch (err) {
      console.warn(`  ⚠ Could not read ${resolved}: ${err.message}`);
    }
  }

  console.log(`\n📚 Knowledge base ready — ${globalKnowledgeBase.length} total entries.\n`);
}

// ============================================================================
//  RAG RETRIEVAL — Keyword Matching
// ============================================================================

/**
 * retrieveContext — Scans globalKnowledgeBase for entries whose Topic or
 * Keywords overlap with words in the user's message.
 *
 * Algorithm:
 *   1. Lowercase the incoming message and split into individual words.
 *   2. For each knowledge-base entry, lowercase its `keywords` and `topic`
 *      fields and check if ANY word from the user message appears in them.
 *   3. Collect all matching Context paragraphs and join them with a
 *      double-newline separator.
 *   4. If nothing matched, return a "no match" sentinel string.
 *
 * @param {string} userMessage – The student's raw question.
 * @returns {string}           – Combined context paragraphs or fallback text.
 */
function retrieveContext(userMessage) {
  // Normalise and tokenise the user message
  const lowerMessage = userMessage.toLowerCase();
  const words = lowerMessage
    .replace(/[^a-záàâãéèêíïóôõúçñ0-9\s]/gi, " ")   // strip punctuation
    .split(/\s+/)                                      // split on whitespace
    .filter((w) => w.length > 2);                      // ignore very short tokens

  const matchedContexts = [];

  for (const entry of globalKnowledgeBase) {
    const topicLower   = entry.topic.toLowerCase();
    const keywordsLower = entry.keywords.toLowerCase();

    // Check if any user word appears in the topic or keywords
    const isMatch = words.some(
      (word) => keywordsLower.includes(word) || topicLower.includes(word)
    );

    if (isMatch && entry.context) {
      matchedContexts.push(entry.context);
    }
  }

  if (matchedContexts.length === 0) {
    return "No specific document match found.";
  }

  // Bundle all matched contexts into one string for the prompt
  return matchedContexts.join("\n\n---\n\n");
}

// ============================================================================
//  EXPRESS APPLICATION
// ============================================================================

const app = express();

// Middleware
app.use(cors());                     // Allow cross-origin requests from the frontend
app.use(express.json());             // Parse incoming JSON bodies

// Serve static assets from our public directory securely
app.use(express.static(path.join(__dirname, '../public')));

// ── HTML delivery endpoints ───────────────────────────────────────────────
app.get("/", (_req, res) => {
  res.sendFile(path.join(__dirname, '../public/index.html'));
});

app.get("/admin", (_req, res) => {
  res.sendFile(path.join(__dirname, '../public/admin.html'));
});

// ── Health-check endpoint (handy for testing) ───────────────────────────────
app.get("/api/health", (_req, res) => {
  res.json({
    status: "online",
    service: "Casper AI API — ICCT Colleges",
    knowledgeBaseEntries: globalKnowledgeBase.length,
  });
});

// ============================================================================
//  POST /api/chat — Main conversational endpoint
// ============================================================================

app.post("/api/chat", async (req, res) => {
  const { studentId, message, history, conversationId } = req.body || {};

  // ── Validate input ──────────────────────────────────────────────────
  if (!message || typeof message !== "string") {
    return res.status(400).json({
      error: "A 'message' string is required in the request body.",
    });
  }

  // ── Determine language from request (sent directly by the frontend) ──
  // The frontend reads the dropdown value and sends it with every chat request.
  // This is more reliable than a DB column since it doesn't require schema changes.
  const ALLOWED_LANGUAGES = ['English', 'Filipino', 'Taglish'];
  const requestedLanguage = req.body?.language;
  const activeLanguage = ALLOWED_LANGUAGES.includes(requestedLanguage) ? requestedLanguage : 'English';
  console.log(`🌐 Language for this request: ${activeLanguage} (from client)`);

  try {

    // ── Feature A (Persistent User Logs) ───────────────────────────────
    await supabase.from('chat_logs').insert([{
      student_id: studentId,
      sender: 'user',
      message_text: message,
      conversation_id: conversationId
    }]);

    // ── Step 1: Compute the incoming message vector ─────────────────────
    const query = message;
    const queryEmbedding = await embedWithRetry(query);
    
    if (!Array.isArray(queryEmbedding) || queryEmbedding.length !== 1536) {
        console.error("❌ Vector generation failed or length is not 1536!");
    }

    // Read local backup file settings (check /tmp as fallback for Render cloud)
    let backupData = {};
    const localFilePath = path.join(__dirname, 'settings.json');
    const tmpFilePath = path.join('/tmp', 'settings.json');
    const settingsReadPath = fs.existsSync(localFilePath) ? localFilePath 
                           : fs.existsSync(tmpFilePath) ? tmpFilePath : null;
    if (settingsReadPath) {
      try {
        backupData = JSON.parse(fs.readFileSync(settingsReadPath, 'utf8'));
      } catch (e) {
        console.error("Failed to parse settings backup file:", e);
      }
    }

    const { data: config, error: configError } = await supabase
      .from('system_settings')
      .select('vector_threshold, active_model, system_instruction')
      .eq('id', 1)
      .single();

    const finalConfig = {
      ...backupData,
      ...(config || {})
    };

    // If there is a saved value, use it. Otherwise, default to -1.0
    const activeThreshold = (finalConfig.vector_threshold !== undefined) 
      ? parseFloat(finalConfig.vector_threshold) 
      : -1.0;

    // Dynamically choose model (or fallback to gemini-2.5-flash if none configured/valid)
    const activeModel = finalConfig.active_model || MODEL_NAME;
      
    console.log(`🤖 Live Vector Processing - Active database threshold rule applied: ${activeThreshold} | Model: ${activeModel}`);

    const { data: dbData, error: dbError } = await supabase.rpc('match_documents', {
        query_embedding: queryEmbedding,
        match_threshold: activeThreshold,
        match_count: 4
    });

    if (dbError) {
        console.error("❌ Supabase RPC Execution Error:", dbError.message);
    } else {
        console.log(`📊 SUCCESS: Vector search returned ${dbData ? dbData.length : 0} row matches from the cloud database!`);
    }

    if (dbData && dbData.length > 0) {
        console.log("🔗 RAW VECTOR SCORES DETECTED:");
        dbData.forEach((row, index) => {
            console.log(`   👉 Match [${index + 1}] | Title: ${row.title} | Similarity Score: ${row.similarity}`);
        });
    }

    const contextText = dbData && dbData.length > 0 
        ? dbData.map(row => row.content).join('\n\n') 
        : "";

    // ── Step 3: Feed context to Gemini or activate fallback ──────────────
    if (!dbData || dbData.length === 0) {
      console.log("⚠️  No vector match — triggering fallback.");

      // Language-aware fallback messages
      const FALLBACK_MESSAGES = {
        'Filipino':  'Hindi ko mahanap ang sagot sa iyong tanong sa aming knowledge base ng unibersidad. Gusto mo bang mag-submit ng opisyal na support ticket sa ICCT Administration?',
        'Taglish':   "Sorry, hindi ko na-find ang sagot sa iyong tanong sa aming university knowledge base. Gusto mo bang mag-submit ng isang opisyal na support ticket sa ICCT Administration?",
        'English':   "I couldn't find an answer to your question in our university knowledge base. Would you like to submit this as an official support ticket to the ICCT Administration?"
      };
      const fallbackReply = FALLBACK_MESSAGES[activeLanguage] || FALLBACK_MESSAGES['English'];

      await supabase.from('chat_logs').insert([{
        student_id: studentId,
        sender: 'bot',
        message_text: fallbackReply,
        conversation_id: conversationId
      }]);

      return res.json({
        reply: fallbackReply,
        offerTicket: true,
        unresolvedInquiry: message
      });
    }
    
    // References using category and title
    const references = dbData.map(chunk => `[Category: ${chunk.category} | Title: ${chunk.title}]`).join(', ');
    console.log(`Matched references: ${references}`);

    const formattedHistory = (history || []).map((entry) => ({
      role: entry.role === "model" ? "model" : "user",
      parts: [{ text: entry.text }],
    }));

    // ── Step 4: Build prompt layout with context ───────────────────────────
    const RAGPrompt = `CONTEXT:
${contextText}

QUESTION:
${message}`;

    const userTurn = {
      role: "user",
      parts: [{ text: RAGPrompt }],
    };

    const contents = [...formattedHistory, userTurn];

    // Build the dynamic system instruction using the active language preference
    const languageRule = activeLanguage === 'English'
      ? `LANGUAGE RULE: Respond in English only.`
      : `CRITICAL LANGUAGE ENFORCEMENT — HIGHEST PRIORITY RULE:
You MUST generate your ENTIRE response in ${activeLanguage} — every word, sentence, and paragraph.
This rule OVERRIDES all other instructions, including the context language below.
- If ${activeLanguage} is "Filipino": Respond fully in Tagalog/Filipino (e.g., "Ang proseso ng..." not "The process of...").
- If ${activeLanguage} is "Taglish": Mix Filipino and English naturally (e.g., "Ang hakbang ay..." with English terms where natural).
DO NOT write even a single sentence in English unless it is a proper noun or acronym (e.g., ICCT, BSIT, SOG).
Even if the CONTEXT below is entirely in English, you MUST translate your answer into ${activeLanguage}.
Failure to respond in ${activeLanguage} is a critical error.`;

    const customHeader = finalConfig.system_instruction;
    let dynamicSystemInstruction = "";
    if (customHeader && customHeader.trim().length > 10) {
      dynamicSystemInstruction = `${languageRule}\n\n${customHeader}`;
    } else {
      dynamicSystemInstruction = `${languageRule}\n\n${getSystemInstruction(activeLanguage)}`;
    }

    // ── Step 5: Call Gemini via the @google/genai SDK ────────────────────
    const response = await generateContentWithRetry(
      activeModel,
      { systemInstruction: dynamicSystemInstruction },
      contents
    );

    let rawReply = (typeof response.text === "string" ? response.text : "").trim();

    // ── Fallback if empty, null, or undefined ─────────────────────────────
    if (!rawReply) {
      console.warn("⚠️ Gemini returned empty/null reply. Defaulting to FALLBACK_TRIGGER.");
      rawReply = "FALLBACK_TRIGGER";
    }

    // Clean response text by removing or escaping control characters (preserving tab, newline, carriage return)
    const cleanAiText = rawReply
      .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]/g, "") 
      .trim();

    console.log(`🤖 Gemini: "${cleanAiText.substring(0, 120)}…"`);

    // ── Feature C (User-Confirmed Ticketing Fallback Pipeline) ────────
    if (cleanAiText.includes("FALLBACK_TRIGGER")) {
      console.log("⚠️  Fallback triggered — prompting for support ticket.");

      const fallbackReply =
        "I couldn't find an answer to your question in our university knowledge base. " +
        "Would you like to submit this as an official support ticket to the ICCT Administration?";

      // Log the bot fallback reply to chat_logs
      await supabase.from('chat_logs').insert([{
        student_id: studentId,
        sender: 'bot',
        message_text: fallbackReply,
        conversation_id: conversationId
      }]);

      return res.status(200).json({ 
          reply: fallbackReply, 
          offerTicket: true,
          unresolvedInquiry: message 
      });
    }

    // ── Feature B (Persistent Bot Logs) ────────────────────────────────
    await supabase.from('chat_logs').insert([{
      student_id: studentId,
      sender: 'bot',
      message_text: cleanAiText,
      conversation_id: conversationId
    }]);

    // ── Normal response — return Gemini's conversational answer ─────────
    return res.status(200).json({
      reply: cleanAiText,
      triggerTicket: false,
    });

  } catch (err) {
    console.error("❌ /api/chat error:", err);

    // Intercept rate limiting (429 / RESOURCE_EXHAUSTED) errors
    const isRateLimit = err.message?.includes("429") || err.message?.includes("RESOURCE_EXHAUSTED") || err.status === 429;
    if (isRateLimit) {
      console.warn("⚠️ Rate limit error handled — returning ticket fallback.");

      // Language-aware fallback messages for rate limits
      const RATE_LIMIT_FALLBACKS = {
        'Filipino': 'Paumanhin, ang AI Assistant ay kasalukuyang nakakaranas ng mataas na dami ng mga tanong at hindi maproseso ang iyong kahilingan sa ngayon. Nais mo bang mag-submit ng isang opisyal na support ticket sa ICCT Administration?',
        'Taglish': 'Sorry, medyo busy ang AI Assistant right now at hindi maproseso ang iyong request. Gusto mo bang mag-submit ng isang opisyal na support ticket sa ICCT Administration na lang?',
        'English': 'The Support Assistant is currently experiencing a very high volume of requests. Would you like to submit your inquiry as an official support ticket to the ICCT Administration instead?'
      };

      const fallbackReply = RATE_LIMIT_FALLBACKS[activeLanguage] || RATE_LIMIT_FALLBACKS['English'];

      try {
        await supabase.from('chat_logs').insert([{
          student_id: studentId || "anonymous",
          sender: 'bot',
          message_text: fallbackReply,
          conversation_id: conversationId
        }]);
      } catch (logErr) {
        console.error("⚠️ Failed to log rate-limit fallback to DB:", logErr.message);
      }

      return res.status(200).json({
        reply: fallbackReply,
        offerTicket: true,
        unresolvedInquiry: message
      });
    }

    return res.status(500).json({
      error: "Internal server error. Please try again later.",
      details: err.message,
    });
  }
});

// ============================================================================
//  STUDENT AUTH ENDPOINT — POST /api/auth/signin
// ============================================================================
//  Accepts { studentId, email } in the request body.
//  • If the student already exists in the 'profiles' table → returns login success.
//  • If the student is new → inserts a new row and returns registration success.
// ============================================================================

app.post("/api/auth/signin", async (req, res) => {
  try {
    const { studentId, password } = req.body;

    // 1. EXTRACT AND VALIDATE PAYLOAD:
    if (!studentId || !password) {
      console.log("⚠️  /api/auth/signin — Missing studentId or password.");
      return res.status(400).json({
        success: false,
        error: "Both 'studentId' and 'password' are required.",
      });
    }

    console.log(`\n🔐 Sign-in gateway attempt — ID: ${studentId}`);

    // 2. ADD THE ADMIN PATTERN INTERCEPTION GATEWAY:
    if (studentId.toUpperCase().startsWith('ADM')) {
      console.log(`🔒 Admin authentication pattern detected for ID: ${studentId}`);
      
      try {
        // Look up the ID using a clean lowercase transformation match
        const searchId = studentId.toLowerCase().trim();
        
        const { data: user, error: dbError } = await supabase
          .from('profiles')
          .select('*')
          .or(`student_id.eq.${searchId},student_id.eq.${searchId.toUpperCase()}`)
          .single();

        if (dbError || !user) {
          console.log(`静态 ⚠️ Admin validation failed. Profile record not found for: ${studentId}`);
          return res.status(401).json({
            success: false,
            error: "Invalid administrator credentials. Access denied."
          });
        }

        // Verify the secure bcrypt password hash
        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
          console.log(`❌ Admin password mismatch for ID: ${studentId}`);
          return res.status(401).json({
            success: false,
            error: "Invalid administrator credentials. Access denied."
          });
        }

        console.log(`👑 Admin authenticated successfully via Database lookup: ${studentId}`);
        return res.json({
          success: true,
          message: "Admin access verified!",
          role: "admin"
        });

      } catch (err) {
        console.error("🔥 Fatal Admin Gateway Error:", err.message);
        return res.status(500).json({ success: false, error: "Internal server gateway error." });
      }
    }

    // 3. FALLBACK TO EXISTING STUDENT LIFECYCLE:
    // ── Check if the student profile matches ID ──────────────────────────────
    const { data: profile, error: selectError } = await supabase
      .from("profiles")
      .select("*")
      .eq("student_id", studentId)
      .maybeSingle();

    if (selectError) {
      console.error("❌ Supabase SELECT error:", selectError.message);
      return res.status(500).json({
        success: false,
        error: "Database query failed.",
        details: selectError.message,
      });
    }

    // ── Verify profile exists and password matches ─────────────────────────
    let isMatch = false;
    if (profile) {
      if (profile.password && (profile.password.startsWith("$2b$") || profile.password.startsWith("$2a$"))) {
        isMatch = await bcrypt.compare(password, profile.password);
      } else {
        // Fallback for legacy plain-text passwords
        isMatch = (profile.password === password);
        
        // Auto-upgrade legacy plain-text password to bcrypt hash on successful sign-in
        if (isMatch) {
          console.log(`🔄 Upgrading legacy plain-text password to bcrypt hash for Student ID: ${studentId}`);
          try {
            const hashed = await bcrypt.hash(password, 10);
            await supabase
              .from("profiles")
              .update({ password: hashed })
              .eq("student_id", studentId);
            profile.password = hashed;
          } catch (updateErr) {
            console.error(`⚠️ Failed to auto-migrate password for Student: ${studentId}`, updateErr.message);
          }
        }
      }
    }

    if (isMatch) {
      console.log(`✅ Student logged in: ${studentId}`);
      return res.json({
        success: true,
        isNewUser: false,
        message: "Student login successful!",
        role: "student",
        profile: profile,
      });
    } else {
      console.log(`⚠️ Login denied for Student ID: ${studentId}`);
      return res.status(401).json({
        success: false,
        error: "Incorrect Student ID or Password. Please try again."
      });
    }

  } catch (err) {
    console.error("❌ /api/auth/signin error:", err);
    return res.status(500).json({
      success: false,
      error: "Internal server error.",
      details: err.message,
    });
  }
});

// ============================================================================
//  STUDENT SIGNUP ENDPOINT — POST /api/auth/signup
// ============================================================================
app.post("/api/auth/signup", async (req, res) => {
  try {
    const { studentId, email, fullName, course, yearLevel, password } = req.body;

    if (!studentId || !email || !fullName || !course || !yearLevel || !password) {
      console.log("⚠️  /api/auth/signup — Missing registration details.");
      return res.status(400).json({
        success: false,
        error: "All registration fields (ID, email, full name, course, year level, password) are required.",
      });
    }

    console.log(`\n🆕 Registration gateway attempt — ID: ${studentId}, Email: ${email}`);

    // Check if the student profile already exists
    const { data: existingProfile, error: selectError } = await supabase
      .from("profiles")
      .select("*")
      .eq("student_id", studentId)
      .maybeSingle();

    if (selectError) {
      console.error("❌ Supabase SELECT error:", selectError.message);
      return res.status(500).json({
        success: false,
        error: "Database check failed.",
      });
    }

    if (existingProfile) {
      return res.status(400).json({
        success: false,
        error: "Account already exists with this Student ID. Please Sign In instead.",
      });
    }

    // Hash the password securely with bcrypt before storing
    const hashedPassword = await bcrypt.hash(password, 10);

    // Execute a clean record insert call directly to our Supabase 'profiles' data table
    const { error: insertError } = await supabase
      .from("profiles")
      .insert([{ 
        student_id: studentId, 
        email: email, 
        full_name: fullName,
        course: course, 
        year_level: yearLevel, 
        password: hashedPassword 
      }]);

    if (insertError) {
      console.error("Supabase Signup Error Detailing:", insertError.message);
      return res.status(500).json({
        success: false,
        error: insertError.message,
      });
    }

    // Retrieve the profile we just created
    const { data: profile, error: fetchError } = await supabase
      .from("profiles")
      .select("*")
      .eq("student_id", studentId)
      .single();

    console.log(`✅ New student registered successfully: ${studentId}`);
    return res.json({
      success: true,
      message: "Student registration successful!",
      profile: profile || { student_id: studentId, email, course, year_level: yearLevel }
    });

  } catch (err) {
    console.error("❌ /api/auth/signup error:", err);
    return res.status(500).json({
      success: false,
      error: "Internal server error.",
      details: err.message,
    });
  }
});

// ============================================================================
//  GET /api/history/:studentId — Retrieve student chat logs history
// ============================================================================
app.get("/api/history/:studentId", async (req, res) => {
  try {
    let query = supabase
      .from("chat_logs")
      .select("*")
      .eq("student_id", req.params.studentId);

    if (req.query.conversationId) {
      query = query.eq("conversation_id", req.query.conversationId);
    }

    const { data, error } = await query.order("created_at", { ascending: true });

    if (error) {
      console.error("❌ Error fetching history:", error.message);
      return res.status(500).json({ error: error.message });
    }

    return res.json(data);
  } catch (err) {
    console.error("❌ /api/history/:studentId error:", err);
    return res.status(500).json({
      error: "Internal server error.",
      details: err.message,
    });
  }
});

// ============================================================================
//  GET /api/conversations/:studentId — Retrieve unique sessions grouped by ID
// ============================================================================
app.get("/api/conversations/:studentId", async (req, res) => {
  try {
    const { data, error } = await supabase
      .from("chat_logs")
      .select("*")
      .eq("student_id", req.params.studentId)
      .eq("is_deleted", false)
      .order("is_pinned", { ascending: false })
      .order("created_at", { ascending: false });

    if (error) {
      console.error("❌ Error fetching conversations:", error.message);
      return res.status(500).json({ error: error.message });
    }

    // Group by conversation_id on the backend to get unique sessions
    const conversationsMap = {};
    data.forEach(log => {
      const convId = log.conversation_id || "default";
      if (!conversationsMap[convId]) {
        conversationsMap[convId] = {
          conversation_id: convId,
          first_message: log.message_text,
          created_at: log.created_at,
          is_pinned: log.is_pinned || false
        };
      } else {
        // Keep scanning to find the chronologically first message for this session
        if (new Date(log.created_at) < new Date(conversationsMap[convId].created_at)) {
          conversationsMap[convId].first_message = log.message_text;
          conversationsMap[convId].created_at = log.created_at;
        }
      }
    });

    const uniqueConversations = Object.values(conversationsMap);
    // Sort unique conversations: pinned first, then by earliest created_at descending (newest session at the top)
    uniqueConversations.sort((a, b) => {
      const pinA = a.is_pinned ? 1 : 0;
      const pinB = b.is_pinned ? 1 : 0;
      if (pinB !== pinA) {
        return pinB - pinA;
      }
      return new Date(b.created_at) - new Date(a.created_at);
    });

    return res.json(uniqueConversations);
  } catch (err) {
    console.error("❌ /api/conversations/:studentId error:", err);
    return res.status(500).json({
      error: "Internal server error.",
      details: err.message,
    });
  }
});

// ============================================================================
//  POST /api/conversations/toggle-pin — Toggle conversation pin state
// ============================================================================
app.post("/api/conversations/toggle-pin", async (req, res) => {
  try {
    const { conversationId, isPinned } = req.body;
    if (!conversationId) {
      return res.status(400).json({ error: "conversationId is required." });
    }

    const { error } = await supabase
      .from("chat_logs")
      .update({ is_pinned: isPinned })
      .eq("conversation_id", conversationId);

    if (error) {
      console.error("❌ Error toggling pin:", error.message);
      return res.status(500).json({ error: error.message });
    }

    return res.json({ success: true, message: "Conversation pin toggled successfully." });
  } catch (err) {
    console.error("❌ /api/conversations/toggle-pin error:", err);
    return res.status(500).json({ error: "Internal server error.", details: err.message });
  }
});

// ============================================================================
//  POST /api/conversations/delete — Soft-delete conversation thread
// ============================================================================
app.post("/api/conversations/delete", async (req, res) => {
  try {
    const { conversationId } = req.body;
    if (!conversationId) {
      return res.status(400).json({ error: "conversationId is required." });
    }

    const { error } = await supabase
      .from("chat_logs")
      .update({ is_deleted: true })
      .eq("conversation_id", conversationId);

    if (error) {
      console.error("❌ Error soft-deleting conversation:", error.message);
      return res.status(500).json({ error: error.message });
    }

    return res.json({ success: true, message: "Conversation soft-deleted successfully." });
  } catch (err) {
    console.error("❌ /api/conversations/delete error:", err);
    return res.status(500).json({ error: "Internal server error.", details: err.message });
  }
});

// ============================================================================
//  GET /api/tickets/:studentId — Retrieve student tickets
// ============================================================================
app.get("/api/tickets/:studentId", async (req, res) => {
  try {
    const { data, error } = await supabase
      .from("tickets")
      .select("*")
      .eq("student_id", req.params.studentId)
      .order("created_at", { ascending: false });

    if (error) {
      console.error("❌ Error fetching tickets:", error.message);
      return res.status(500).json({ error: error.message });
    }

    return res.json(data);
  } catch (err) {
    console.error("❌ /api/tickets/:studentId error:", err);
    return res.status(500).json({
      error: "Internal server error.",
      details: err.message,
    });
  }
});

// ============================================================================
//  GET /api/student/tickets/:studentId — Student Ticket Tracking Center
// ============================================================================
app.get("/api/student/tickets/:studentId", async (req, res) => {
  try {
    const studentId = req.params.studentId;
    const { data, error } = await supabase
      .from("tickets")
      .select("*")
      .eq("student_id", studentId)
      .order("created_at", { ascending: false });

    if (error) {
      console.error("❌ Error fetching student tickets:", error.message);
      return res.status(500).json({ error: error.message });
    }

    return res.status(200).json({ tickets: data });
  } catch (err) {
    console.error("❌ /api/student/tickets/:studentId error:", err);
    return res.status(500).json({
      error: "Internal server error.",
      details: err.message,
    });
  }
});

// ============================================================================
//  POST /api/student/settings — Save student language preference
// ============================================================================
//  Accepts { studentId, language } — updates system_language in the profiles table.
//  The /api/chat route reads this field fresh on every request, so the change
//  takes effect on the very next message the student sends.
// ============================================================================
app.post("/api/student/settings", async (req, res) => {
  try {
    const { studentId, language } = req.body;

    const ALLOWED_LANGUAGES = ["English", "Filipino", "Taglish"];

    if (!studentId || !language) {
      return res.status(400).json({ success: false, message: "Missing studentId or language." });
    }

    if (!ALLOWED_LANGUAGES.includes(language)) {
      return res.status(400).json({ success: false, message: `Invalid language. Must be one of: ${ALLOWED_LANGUAGES.join(", ")}.` });
    }

    const { error } = await supabase
      .from("profiles")
      .update({ system_language: language })
      .eq("student_id", studentId);

    if (error) {
      console.error("❌ Failed to update student language:", error.message);
      return res.status(500).json({ success: false, message: error.message });
    }

    console.log(`🌐 Language preference updated — Student: ${studentId} → ${language}`);
    return res.status(200).json({ success: true, message: `Language updated to ${language}.` });

  } catch (err) {
    console.error("❌ /api/student/settings error:", err);
    return res.status(500).json({ success: false, message: "Internal server error.", details: err.message });
  }
});

// ============================================================================
//  POST /api/student/profile — Update student program and year level
// ============================================================================
//  Accepts { studentId, course, yearLevel } — updates the profiles table.
// ============================================================================
app.post("/api/student/profile", async (req, res) => {
  try {
    const { studentId, course, yearLevel } = req.body;

    if (!studentId) {
      return res.status(400).json({ success: false, message: "Missing studentId." });
    }

    const updates = {};
    if (course !== undefined && course !== null && course.trim() !== '') {
      updates.course = course.trim();
    }
    if (yearLevel !== undefined && yearLevel !== null && yearLevel.trim() !== '') {
      updates.year_level = yearLevel.trim();
    }

    if (Object.keys(updates).length === 0) {
      return res.status(400).json({ success: false, message: "No valid fields to update." });
    }

    const { error } = await supabase
      .from("profiles")
      .update(updates)
      .eq("student_id", studentId);

    if (error) {
      console.error("❌ Failed to update student profile:", error.message);
      return res.status(500).json({ success: false, message: error.message });
    }

    console.log(`👤 Profile updated — Student: ${studentId}`, updates);
    return res.status(200).json({ success: true, message: "Profile updated successfully." });

  } catch (err) {
    console.error("❌ /api/student/profile error:", err);
    return res.status(500).json({ success: false, message: "Internal server error.", details: err.message });
  }
});

// ============================================================================
//  POST /api/tickets/submit — Submit a user-confirmed support ticket (Max 3 Open)
// ============================================================================
app.post("/api/tickets/submit", async (req, res) => {
  try {
    const studentId = req.body.student_id || req.body.studentId; // Read the active student's session ID (supports both formats)
    const unresolvedInquiry = req.body.unresolvedInquiry || req.body.unresolved_inquiry;

    if (!studentId || !unresolvedInquiry) {
      return res.status(400).json({
        success: false,
        message: "Missing studentId or unresolvedInquiry in request body."
      });
    }

    console.log(`🎫 Submission request for student: ${studentId}`);

    // A. Get the maximum allowed tickets rule set by the admin
    const { data: config } = await supabase
      .from('system_settings')
      .select('max_tickets')
      .eq('id', 1)
      .single();
    
    const allowedMax = config ? parseInt(config.max_tickets) : 3;

    // B. Count how many active, pending tickets this student already has in the system
    const { count: currentActiveTickets, error: countError } = await supabase
      .from('tickets')
      .select('*', { count: 'exact', head: true })
      .eq('student_id', studentId)
      .or('status.eq.Pending,status.eq.pending'); // Case-insensitive check to ensure correct counting

    if (countError) {
      console.error("❌ Error counting existing tickets:", countError.message);
      return res.status(500).json({ success: false, message: "Database query failed.", details: countError.message });
    }

    // C. The Security Interception Gate:
    if (currentActiveTickets >= allowedMax) {
      console.log(`⚠️ Ticket Blocked: Student ${studentId} reached their administrative limit of ${allowedMax}`);
      return res.status(400).json({
        success: false,
        error: "Ticket limit reached.",
        message: `You have reached your maximum limit of ${allowedMax} pending ticket(s). Please wait for administration to resolve your open cases.`
      });
    }

    // 2. IF THE GATE PASSES, CONTINUE TO INSERT THE TICKET:
    let computedPriority = "Medium";
    const lowUrgencyWords = ["hi", "hello", "tanong", "paki-check"];
    const highUrgencyWords = ["bagsak", "mali", "error", "overdue", "scholarship", "system", "blocked", "banned"];
    const contentLower = unresolvedInquiry.toLowerCase();
    
    if (highUrgencyWords.some(word => contentLower.includes(word))) {
        computedPriority = "High";
    } else if (lowUrgencyWords.some(word => contentLower.includes(word))) {
        computedPriority = "Low";
    }

    const { error: insertError } = await supabase
      .from('tickets')
      .insert([{
        student_id: studentId,
        unresolved_inquiry: unresolvedInquiry,
        status: 'Pending',
        priority: computedPriority
      }]);

    if (insertError) {
      console.error("❌ Error creating support ticket:", insertError.message);
      return res.status(500).json({ success: false, message: "Failed to submit ticket.", details: insertError.message });
    }

    return res.json({ success: true, message: "Support ticket successfully filed with the ICCT Administration!" });

  } catch (err) {
    console.error("Ticket submission validation error:", err.message);
    return res.status(500).json({ success: false, error: "Internal service error." });
  }
});


// ============================================================================
//  GET /api/admin/settings — Retrieve Admin configurations from Supabase
// ============================================================================
app.get('/api/admin/settings', async (req, res) => {
  try {
    // Read local backup if it exists (check /tmp as fallback for Render cloud)
    let backupData = {};
    const localFilePath = path.join(__dirname, 'settings.json');
    const tmpFilePath = path.join('/tmp', 'settings.json');
    const settingsReadPath = fs.existsSync(localFilePath) ? localFilePath
                           : fs.existsSync(tmpFilePath) ? tmpFilePath : null;
    if (settingsReadPath) {
      try {
        backupData = JSON.parse(fs.readFileSync(settingsReadPath, 'utf8'));
      } catch (e) {
        console.error("Failed to parse settings backup file:", e);
      }
    }

    const { data, error } = await supabase
      .from('system_settings')
      .select('*')
      .eq('id', 1)
      .single();
      
    if (error) {
      console.warn("⚠️ Failed to load settings from Supabase database. Falling back to local file:", error.message);
      return res.json({ success: true, settings: backupData });
    }

    // Merge database configurations and file backup configurations
    const mergedSettings = {
      ...backupData,
      ...data
    };

    return res.json({ success: true, settings: mergedSettings });
  } catch (err) {
    console.error("Error fetching settings:", err.message);
    return res.status(500).json({ success: false, error: err.message });
  }
});

// ============================================================================
//  POST /api/admin/settings — Save Admin configurations to Supabase
// ============================================================================
app.post('/api/admin/settings', async (req, res) => {
  const { vector_threshold, max_tickets, active_model, system_instruction } = req.body;
  try {
    // Save locally to server first as a reliable backup.
    // Use /tmp on cloud environments (e.g. Render) where the app directory is read-only.
    const backupSettings = {
      vector_threshold: parseFloat(vector_threshold),
      max_tickets: parseInt(max_tickets),
      active_model,
      system_instruction
    };
    const localSettingsPath = fs.existsSync(path.join(__dirname, 'settings.json'))
      ? path.join(__dirname, 'settings.json')
      : path.join('/tmp', 'settings.json');
    try {
      fs.writeFileSync(localSettingsPath, JSON.stringify(backupSettings, null, 2));
    } catch (writeErr) {
      // If both fail (permissions), write to /tmp as absolute fallback
      try {
        fs.writeFileSync(path.join('/tmp', 'settings.json'), JSON.stringify(backupSettings, null, 2));
      } catch (tmpErr) {
        console.warn('⚠️ Could not write settings backup file:', tmpErr.message);
      }
    }

    // Save to Supabase database (primary store)
    const { error } = await supabase
      .from('system_settings')
      .upsert({ 
        id: 1, 
        vector_threshold: parseFloat(vector_threshold), 
        max_tickets: parseInt(max_tickets),
        active_model: active_model,
        system_instruction: system_instruction
      });
      
    if (error) {
      console.warn("⚠️ Database settings upsert failed:", error.message);
      return res.status(500).json({ success: false, error: `Database error: ${error.message}` });
    }
    
    console.log(`⚙️ System configs saved successfully! Threshold: ${vector_threshold} | Model: ${active_model}`);
    return res.json({ success: true, message: "Settings saved successfully!" });
  } catch (err) {
    console.error("Error saving settings:", err.message);
    return res.status(500).json({ success: false, error: err.message });
  }
});


// ============================================================================
//  GET /api/admin/analytics — System Metrics & Chart Aggregation
// ============================================================================
app.get("/api/admin/analytics", async (req, res) => {
  try {
    // 1. Fetch chat logs to determine total inquiries and categorize them
    const { data: chatLogs, error: logsError } = await supabase
      .from("chat_logs")
      .select("*");

    if (logsError) {
      console.error("❌ Analytics error fetching chat logs:", logsError.message);
      return res.status(500).json({ error: logsError.message });
    }

    // Categorize inquiries based on message keywords
    const categories = {
      Enrollment: 0,
      Tuition: 0,
      "Document Requests": 0,
      Scheduling: 0
    };

    chatLogs.forEach(log => {
      const text = (log.message_text || "").toLowerCase();
      if (text.includes("enroll") || text.includes("admission") || text.includes("apply") || text.includes("requirements")) {
        categories.Enrollment++;
      } else if (text.includes("tuition") || text.includes("payment") || text.includes("fee") || text.includes("fees") || text.includes("bayad") || text.includes("scholarship") || text.includes("discount")) {
        categories.Tuition++;
      } else if (text.includes("document") || text.includes("documents") || text.includes("sog") || text.includes("transcript") || text.includes("tor") || text.includes("diploma") || text.includes("certificate")) {
        categories["Document Requests"]++;
      } else if (text.includes("schedule") || text.includes("scheduling") || text.includes("sched") || text.includes("calendar") || text.includes("class") || text.includes("classes") || text.includes("subject") || text.includes("semester")) {
        categories.Scheduling++;
      }
    });

    // 2. Fetch tickets count (total pending vs resolved)
    const { data: tickets, error: ticketsError } = await supabase
      .from("tickets")
      .select("*");

    if (ticketsError) {
      console.error("❌ Analytics error fetching tickets:", ticketsError.message);
      return res.status(500).json({ error: ticketsError.message });
    }

    let pendingTickets = 0;
    let resolvedTickets = 0;

    tickets.forEach(ticket => {
      if (ticket.status === "Pending") {
        pendingTickets++;
      } else if (ticket.status === "Resolved") {
        resolvedTickets++;
      }
    });

    // 3. Calculate resolution rate
    const totalInquiries = chatLogs.length;
    const totalTickets = tickets.length;
    const resolutionRate = totalInquiries > 0 ? ((totalInquiries - totalTickets) / totalInquiries) * 100 : 100;

    return res.json({
      totalInquiries,
      pendingTickets,
      resolutionRate,
      chartData: {
        categories,
        ticketStatus: {
          Pending: pendingTickets,
          Resolved: resolvedTickets
        }
      }
    });

  } catch (err) {
    console.error("❌ /api/admin/analytics error:", err);
    return res.status(500).json({
      error: "Internal server error.",
      details: err.message
    });
  }
});

// ============================================================================
//  GET /api/admin/stats — Live Administrative Statistics
// ============================================================================
app.get("/api/admin/stats", async (req, res) => {
  try {
    // 1. QUERY LIVE TICKET COUNTS
    const { data: tickets, error: ticketsError } = await supabase
      .from("tickets")
      .select("status");

    if (ticketsError) {
      console.error("❌ Stats error fetching tickets:", ticketsError.message);
      return res.status(500).json({ error: ticketsError.message });
    }

    const totalTickets = tickets?.length ?? 0;
    const pendingTickets = tickets ? tickets.filter(t => t.status?.toLowerCase() === "pending").length : 0;

    const ticketStatusDist = {
      Pending: 0,
      Resolved: 0
    };
    if (tickets) {
      tickets.forEach(t => {
        const status = t.status;
        if (status) {
          const key = status.charAt(0).toUpperCase() + status.slice(1).toLowerCase();
          ticketStatusDist[key] = (ticketStatusDist[key] || 0) + 1;
        }
      });
    }

    // 2. FETCH CATEGORIES + TITLES LIVE FROM school_knowledge
    //    We pull titles too so keyword matching covers what students actually ask about,
    //    not just the bare category name.
    const { data: knowledgeRows, error: knowledgeError } = await supabase
      .from("school_knowledge")
      .select("category, title");

    if (knowledgeError) {
      console.error("❌ Stats error fetching knowledge categories:", knowledgeError.message);
      return res.status(500).json({ error: knowledgeError.message });
    }

    // Group titles under each unique category  { "Admission": ["SHS Pre Reg...", ...], ... }
    const categoryTitleMap = {};
    for (const row of (knowledgeRows || [])) {
      const cat = (row.category || "Unassigned").trim();
      if (!categoryTitleMap[cat]) categoryTitleMap[cat] = [];
      if (row.title) categoryTitleMap[cat].push(row.title);
    }
    const distinctCategories = Object.keys(categoryTitleMap);

    // 3. FETCH CHAT LOGS FOR KEYWORD COUNTING
    const { data: chatLogs, error: logsError } = await supabase
      .from("chat_logs")
      .select("message_text")
      .eq("sender", "user");

    if (logsError) {
      console.error("❌ Stats error fetching chat logs:", logsError.message);
      return res.status(500).json({ error: logsError.message });
    }

    // 4. COUNT MATCHES PER CATEGORY — each message assigned to exactly ONE category.
    //    We sort categories by their keyword specificity (descending) so the most specific
    //    category wins when a message could match multiple.
    //    This guarantees: sum(categoryCounts) == messages that matched ≥1 category ≤ totalInquiries
    const logTexts = (chatLogs || []).map(l => (l.message_text || "").toLowerCase());
    const stopWords = new Set([
      "and", "or", "the", "for", "of", "a", "an", "in", "on", "to",
      "with", "are", "is", "its", "at", "by", "be", "as", "from", "about"
    ]);

    function extractStems(text) {
      return text
        .toLowerCase()
        .split(/[\s/,&\-:()]+/)
        .map(w => w.replace(/[^a-z0-9]/g, ""))
        .filter(w => w.length > 2 && !stopWords.has(w))
        .map(w => w.slice(0, Math.min(6, w.length)));
    }

    // Pre-compute keyword sets per category
    const categoryKeywords = {};
    for (const cat of distinctCategories) {
      const titles = categoryTitleMap[cat] || [];
      const allSourceText = [cat, ...titles].join(" ");
      categoryKeywords[cat] = [...new Set(extractStems(allSourceText))];
    }

    // Sort categories by number of keywords descending (more specific first)
    const sortedCategories = [...distinctCategories].sort(
      (a, b) => categoryKeywords[b].length - categoryKeywords[a].length
    );

    // Assign each message to the FIRST matching category only.
    const categoryCountMap = {};
    for (const cat of distinctCategories) categoryCountMap[cat] = 0;

    for (const text of logTexts) {
      for (const cat of sortedCategories) {
        const keywords = categoryKeywords[cat];
        if (keywords.length > 0 && keywords.some(kw => text.includes(kw))) {
          categoryCountMap[cat]++;
          break; // ← stop at first match — no double counting
        }
      }
    }

    // 5. AI RESOLUTION MATH
    const totalInquiries = chatLogs?.length ?? 0;
    const selfServed = Math.max(0, totalInquiries - totalTickets);
    const resolutionRateVal = totalInquiries > 0 ? (selfServed / totalInquiries) * 100 : 100.0;
    const resolutionRate = `${resolutionRateVal.toFixed(1)}%`;

    // 6. BUILD RESPONSE ARRAY — sorted descending by count, all categories included
    //    Uses { category, count } shape so the frontend maps item.category / item.count
    const categoriesPayload = Object.entries(categoryCountMap)
      .map(([category, count]) => ({ category, count }))
      .sort((a, b) => b.count - a.count);

    // Legacy flat object (kept for any other consumers that read data.categories)
    const categories = Object.fromEntries(
      categoriesPayload.map(({ category, count }) => [category, count])
    );

    return res.status(200).json({
      totalInquiries,
      pendingTickets,
      resolutionRate,
      categories,           // legacy flat map
      categoriesArray: categoriesPayload, // dynamic array: { category, count }
      ticketStatusDist
    });

  } catch (err) {
    console.error("❌ /api/admin/stats error:", err);
    return res.status(500).json({
      error: "Internal server error.",
      details: err.message
    });
  }
});

// ============================================================================
//  GET /api/admin/tickets — Global Tickets Fetch (Pending first)
// ============================================================================
app.get("/api/admin/tickets", async (req, res) => {
  try {
    const { data: tickets, error } = await supabase
      .from("tickets")
      .select("*")
      .order("created_at", { ascending: false });

    if (error) {
      console.error("❌ Error fetching global tickets:", error.message);
      return res.status(500).json({ error: error.message });
    }

    if (!tickets || tickets.length === 0) {
      return res.json([]);
    }

    // Extract unique student IDs
    const studentIds = [...new Set(tickets.map(t => t.student_id))].filter(Boolean);
    let profilesMap = {};

    if (studentIds.length > 0) {
      const { data: profiles, error: profilesError } = await supabase
        .from("profiles")
        .select("student_id, email, full_name, course, year_level")
        .in("student_id", studentIds);

      if (profilesError) {
        console.error("⚠️ Error fetching student profiles for tickets:", profilesError.message);
      } else if (profiles) {
        profiles.forEach(p => {
          profilesMap[p.student_id] = p;
        });
      }
    }

    // Merge profile data with ticket details
    const ticketsWithProfiles = tickets.map(t => ({
      ...t,
      student_email: profilesMap[t.student_id]?.email || "N/A",
      student_name: profilesMap[t.student_id]?.full_name || "N/A",
      student_course: profilesMap[t.student_id]?.course || "N/A",
      student_year_level: profilesMap[t.student_id]?.year_level || "N/A"
    }));

    return res.json(ticketsWithProfiles);
  } catch (err) {
    console.error("❌ /api/admin/tickets error:", err);
    return res.status(500).json({
      error: "Internal server error.",
      details: err.message
    });
  }
});

// ============================================================================
//  POST/PUT /api/admin/tickets/resolve — Resolve a Ticket
// ============================================================================
app.post("/api/admin/tickets/resolve", async (req, res) => {
  try {
    const { ticketId } = req.body;
    if (!ticketId) {
      return res.status(400).json({ error: "ticketId is required inside request body." });
    }

    const { error } = await supabase
      .from("tickets")
      .update({ status: "Resolved" })
      .eq("id", ticketId);

    if (error) {
      console.error("❌ Error resolving ticket:", error.message);
      return res.status(500).json({ error: error.message });
    }

    return res.json({
      success: true,
      message: "Ticket successfully marked as Resolved!"
    });
  } catch (err) {
    console.error("❌ /api/admin/tickets/resolve error:", err);
    return res.status(500).json({
      error: "Internal server error.",
      details: err.message
    });
  }
});

// ============================================================================
//  PUT /api/admin/tickets/:id — Resolve a Ticket with Admin Reply
// ============================================================================
app.put("/api/admin/tickets/:id", async (req, res) => {
  try {
    const { id } = req.params;
    const { status, admin_reply } = req.body;

    const finalStatus = (status && status.toLowerCase() === "resolved") ? "Resolved" : (status || "Resolved");

    const { error } = await supabase
      .from("tickets")
      .update({ 
        status: finalStatus, 
        admin_reply: admin_reply 
      })
      .eq("id", id);

    if (error) {
      console.error(`❌ Error updating ticket ID ${id}:`, error.message);
      return res.status(500).json({ error: error.message });
    }

    return res.status(200).json({ success: true });

  } catch (err) {
    console.error(`❌ /api/admin/tickets/:id error:`, err);
    return res.status(500).json({
      error: "Internal server error.",
      details: err.message
    });
  }
});


// ============================================================================
//  ADMIN AUTH ENDPOINT — POST /api/auth/admin-login
// ============================================================================
app.post("/api/auth/admin-login", (req, res) => {
  try {
    const { masterKey } = req.body;
    if (!masterKey) {
      console.log("⚠️ Admin login attempt — missing masterKey.");
      return res.status(400).json({ error: "Master key is required in request body." });
    }

    const defaultKey = "admin123";
    const envKey = process.env.ADMIN_MASTER_KEY;

    if (masterKey === defaultKey || (envKey && masterKey === envKey)) {
      console.log("🔒 Admin login successful.");
      return res.json({ success: true, message: "Admin authenticated successfully." });
    } else {
      console.log("⚠️ Admin login failed — invalid master key.");
      return res.status(401).json({ error: "Invalid administrative master access key." });
    }
  } catch (err) {
    console.error("❌ /api/auth/admin-login error:", err);
    return res.status(500).json({ error: "Internal server error." });
  }
});

// ============================================================================
//  POST /api/admin/knowledge — Add New Knowledge Base Entry
// ============================================================================
app.post("/api/admin/knowledge", async (req, res) => {
  try {
    const { category, title, content } = req.body;

    // ── Validate input ──────────────────────────────────────────────────
    if (!category || !title || !content) {
      return res.status(400).json({
        success: false,
        message: "All fields are required: 'category', 'title', and 'content'.",
      });
    }

    console.log(`\n📝 Admin knowledge creation — Category: "${category}", Title: "${title}"`);

    // ── Step 1: Generate 1536-dim embedding via Gemini ───────────────────
    const embedding = await embedWithRetry(content);
    console.log(`  ✓ Embedding generated (${embedding.length} dimensions)`);

    // ── Step 2: Insert into Supabase school_knowledge table ──────────────
    const { error: insertError } = await supabase
      .from("school_knowledge")
      .insert({
        category,
        title,
        content,
        embedding,
      });

    if (insertError) {
      console.error("❌ Supabase INSERT error:", insertError.message);
      return res.status(500).json({
        success: false,
        message: "Failed to save knowledge entry to database.",
        details: insertError.message,
      });
    }

    console.log(`  ✅ Knowledge entry saved: "${title}"`);
    return res.status(200).json({
      success: true,
      message: "Knowledge base updated successfully!",
    });

  } catch (err) {
    console.error("❌ /api/admin/knowledge error:", err);
    return res.status(500).json({
      success: false,
      message: "Internal server error.",
      details: err.message,
    });
  }
});

// ============================================================================
//  GET /api/admin/knowledge — List All Knowledge Base Entries
// ============================================================================
app.get("/api/admin/knowledge", async (req, res) => {
  try {
    const { data, error } = await supabase
      .from("school_knowledge")
      .select("id, category, title, content")
      .order("id", { ascending: false });

    if (error) {
      console.error("❌ Supabase SELECT error:", error.message);
      return res.status(500).json({ success: false, message: error.message });
    }

    return res.status(200).json(data || []);

  } catch (err) {
    console.error("❌ /api/admin/knowledge GET error:", err);
    return res.status(500).json({ success: false, message: "Internal server error." });
  }
});

// ============================================================================
//  DELETE /api/admin/knowledge/:id — Remove Knowledge Base Entry
// ============================================================================
app.delete("/api/admin/knowledge/:id", async (req, res) => {
  try {
    const { id } = req.params;

    console.log(`\n🗑️  Admin knowledge deletion — Record ID: ${id}`);

    const { error } = await supabase
      .from("school_knowledge")
      .delete()
      .eq("id", id);

    if (error) {
      console.error("❌ Supabase DELETE error:", error.message);
      return res.status(500).json({
        success: false,
        message: "Failed to delete knowledge entry from database.",
        details: error.message,
      });
    }

    console.log(`  ✅ Knowledge entry deleted — ID: ${id}`);
    return res.status(200).json({
      success: true,
      message: "Policy deleted successfully!",
    });

  } catch (err) {
    console.error("❌ /api/admin/knowledge/:id DELETE error:", err);
    return res.status(500).json({
      success: false,
      message: "Internal server error.",
      details: err.message,
    });
  }
});

// ============================================================================
//  PUT /api/admin/knowledge/:id — Update Knowledge Base Entry metadata
// ============================================================================
app.put("/api/admin/knowledge/:id", async (req, res) => {
  try {
    const { id } = req.params;
    const { category, title, content } = req.body;

    if (!category || !title || !content) {
      return res.status(400).json({
        success: false,
        message: "All fields are required: 'category', 'title', and 'content'.",
      });
    }

    console.log(`\n📝 Admin knowledge update — ID: ${id}, Category: "${category}", Title: "${title}"`);

    // runs a clean update call to Supabase
    const { data, error } = await supabase
      .from("school_knowledge")
      .update({ category, title, content })
      .eq("id", id);

    if (error) {
      console.error("❌ Supabase UPDATE error:", error.message);
      return res.status(500).json({
        success: false,
        message: "Failed to update knowledge entry in database.",
        details: error.message,
      });
    }

    return res.status(200).json({ success: true });
  } catch (err) {
    console.error("❌ /api/admin/knowledge/:id PUT error:", err);
    return res.status(500).json({ success: false, message: "Internal server error." });
  }
});

// ============================================================================
//  POST /api/admin/knowledge/:id/revectorize — Generate & update vector embedding
// ============================================================================
app.post("/api/admin/knowledge/:id/revectorize", async (req, res) => {
  try {
    const { id } = req.params;
    console.log(`\n🌀 Re-vectorizing knowledge entry — ID: ${id}`);

    // Fetch current content from Supabase
    const { data, error: selectError } = await supabase
      .from("school_knowledge")
      .select("content")
      .eq("id", id)
      .single();

    if (selectError || !data) {
      console.error("❌ Error fetching content for re-vectorization:", selectError?.message);
      return res.status(404).json({ success: false, message: "Entry not found." });
    }

    // Generate fresh embedding via Gemini model
    const embedding = await embedWithRetry(data.content);
    console.log(`  ✓ Fresh embedding generated (${embedding.length} dimensions)`);

    // Update in Supabase
    const { error: updateError } = await supabase
      .from("school_knowledge")
      .update({ embedding })
      .eq("id", id);

    if (updateError) {
      console.error("❌ Supabase EMBEDDING update error:", updateError.message);
      return res.status(500).json({
        success: false,
        message: "Failed to save embedding to database.",
        details: updateError.message,
      });
    }

    console.log(`  ✅ Re-vectorization success for ID: ${id}`);
    return res.status(200).json({ success: true, message: "Embedding updated successfully." });
  } catch (err) {
    console.error("❌ Re-vectorization error:", err);
    return res.status(500).json({ success: false, message: "Internal server error." });
  }
});

// ============================================================================
//  SERVER BOOT
// ============================================================================

console.log("\n🚀 Casper AI Backend — ICCT Colleges Capstone");
console.log("──────────────────────────────────────────");
console.log("Loading knowledge base CSV files…\n");

loadCSVFiles();

console.log(`\n🗄️  Supabase connected → ${supabaseUrl}`);

app.listen(PORT, () => {
  console.log(`✅ Server running at http://localhost:${PORT}`);
  console.log(`   POST /api/chat                    — Chat endpoint`);
  console.log(`   POST /api/auth/signin             — Student authentication`);
  console.log(`   POST /api/tickets/submit          — Submit support ticket`);
  console.log(`   GET  /api/history/:studentId       — Retrieve chat history`);
  console.log(`   GET  /api/conversations/:studentId — Retrieve unique chat sessions`);
  console.log(`   GET  /api/tickets/:studentId       — Retrieve support tickets`);
  console.log(`   GET  /                            — Health check\n`);
});
