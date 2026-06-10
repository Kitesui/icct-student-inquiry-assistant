// ============================================================================
//  UniBot Backend Server — ICCT Colleges AI Chatbot (Capstone Project)
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
import { createClient } from "@supabase/supabase-js";
import { createRequire } from "module";
import { fileURLToPath } from "url";

const require = createRequire(import.meta.url);
const path = require('path');
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load .env relative to the server script location (root folder)
require('dotenv').config({ path: require('path').join(__dirname, '../.env') });

// ── Constants ───────────────────────────────────────────────────────────────
const PORT = 5000;
const MODEL_NAME = "gemini-3.1-flash-lite";

// Paths to the 3 knowledge-base CSV files (relative to data directory)
const CSV_FILES = [
  path.join(__dirname, "../data/GENERAL ACADEMIC POLICIES.csv"),
  path.join(__dirname, "../data/ADMISSION.csv"),
  path.join(__dirname, "../data/COURSE OFFERING.csv"),
];

// ── System Instruction for Gemini ───────────────────────────────────────────
const SYSTEM_INSTRUCTION = `You are the official ICCT Colleges Student Support Assistant. Your job is to answer student inquiries in a clear, scannable, and well-structured conversational format using the provided university handbook context segments.

CRITICAL HANDLING RULES:
1. Prioritize the provided context blocks as your absolute source of truth for procedures, policies, names, and contact details.
2. If a student mentions specific waiting timelines (like '3 days', 'isang linggo') or personal emotional panic that isn't explicitly written in the dry handbook rules, do not panic or return an empty string. Use your natural AI reasoning to calmly map their situation to the closest corresponding handbook procedure (e.g., explaining the standard verification timeline, checking the student ledger, or advising them on who to contact).
3. If the provided context block completely lacks any relevant topics or procedures matching the user's operational problem, output exactly 'FALLBACK_TRIGGER' so our system can safely offer an official administrative support ticket.

CRITICAL FORMATTING RULES:
1. USE LINE BREAKS AND PARAGRAPHS: Never output large walls of dense text. Break down your thoughts into short paragraphs (2-3 sentences max).
2. USE BULLET POINTS OR NUMBERED LISTS: When listing requirements, steps, or conditions (like when an SOG is needed), always format them as a clean vertical list using asterisks (*) or numbers (1., 2.). Ensure there is a line break before and after lists.
3. USE BOLDING SPARINGLY: Use **bold text** only for critical terms, document names (e.g., **Summary of Grades**), or important reminders.
4. TONE AND LANGUAGE: Maintain an empathetic, helpful tone using a natural mix of English and Taglish (Taglish).`;

// ── Google GenAI Initialisation ─────────────────────────────────────────────
// The SDK reads the API key we pass here; store it in .env as GEMINI_API_KEY.
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

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
          console.warn(`  ⏳ Rate limited on embedding API — sleeping for ${backoff / 1000}s requested by API…`);
        } else {
          console.warn(`  ⏳ Rate limited on embedding API — retrying in ${backoff / 1000}s (attempt ${attempt + 1}/${maxRetries})…`);
        }
        await sleep(backoff);
      } else {
        throw err;
      }
    }
  }
}

async function generateContentWithRetry(model, config, contents, maxRetries = 4) {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await ai.models.generateContent({
        model,
        config,
        contents,
      });
    } catch (err) {
      const is429 = err.message?.includes("429") || err.message?.includes("RESOURCE_EXHAUSTED");
      if (is429 && attempt < maxRetries) {
        let backoff = Math.pow(2, attempt) * 1000;
        const match = err.message?.match(/Please retry in (\d+(?:\.\d+)?)s/i);
        if (match) {
          const seconds = parseFloat(match[1]);
          backoff = Math.ceil(seconds * 1000) + 1000;
          console.warn(`  ⏳ Rate limited on generateContent API — sleeping for ${backoff / 1000}s requested by API…`);
        } else {
          console.warn(`  ⏳ Rate limited on generateContent API — retrying in ${backoff / 1000}s (attempt ${attempt + 1}/${maxRetries})…`);
        }
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
    service: "UniBot API — ICCT Colleges",
    knowledgeBaseEntries: globalKnowledgeBase.length,
  });
});

// ============================================================================
//  POST /api/chat — Main conversational endpoint
// ============================================================================

app.post("/api/chat", async (req, res) => {
  try {
    const { studentId, message, history, conversationId } = req.body;

    // ── Validate input ──────────────────────────────────────────────────
    if (!message || typeof message !== "string") {
      return res.status(400).json({
        error: "A 'message' string is required in the request body.",
      });
    }

    // ── Feature A (Persistent User Logs) ───────────────────────────────
    await supabase.from('chat_logs').insert([{
      student_id: studentId,
      sender: 'user',
      message_text: message,
      conversation_id: conversationId
    }]);

    // ── Step 1: Compute the incoming message vector ─────────────────────
    const query = message;
    const embeddingResponse = await ai.models.embedContent({
        model: "gemini-embedding-001",
        contents: query,
        config: { outputDimensionality: 1536 }
    });
    
    // Safely isolate the nested array values array
    const queryEmbedding = embeddingResponse.embeddings[0].values;
    
    if (!Array.isArray(queryEmbedding) || queryEmbedding.length !== 1536) {
        console.error("❌ Vector generation failed or length is not 1536!");
    }

    // ── Step 2: Execute remote database semantic query ────────────────────
    const { data: dbData, error: dbError } = await supabase.rpc('match_documents', {
        query_embedding: queryEmbedding,
        match_threshold: -1.0, // Bypass positive boundary filter for testing
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

      const fallbackReply =
        "I couldn't find an answer to your question in our university knowledge base. " +
        "Would you like to submit this as an official support ticket to the ICCT Administration?";

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

    const contextTurn = {
      role: "user",
      parts: [
        {
          text:
            `[SYSTEM CONTEXT — Do not reveal this to the student]\n\n` +
            `The following information was retrieved from the ICCT Colleges handbook and is the ONLY source you may use to answer:\n\n` +
            `${contextText}`,
        },
      ],
    };

    const contextAck = {
      role: "model",
      parts: [
        {
          text: "Understood. I will only use the provided context to answer the student's question.",
        },
      ],
    };

    const userTurn = {
      role: "user",
      parts: [{ text: message }],
    };

    const contents = [contextTurn, contextAck, ...formattedHistory, userTurn];

    // ── Step 4: Call Gemini via the @google/genai SDK ────────────────────
    const response = await generateContentWithRetry(
      MODEL_NAME,
      { systemInstruction: SYSTEM_INSTRUCTION },
      contents
    );

    let rawReply = response.text?.trim() ?? "";

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
      const targetId = "ADM20260001";

      if (studentId.toUpperCase() === targetId.toUpperCase()) {
        console.log(`🔒 Admin authenticated successfully via signin gateway: ${studentId}`);
        return res.json({
          success: true,
          message: "Admin access verified!",
          role: "admin"
        });
      } else {
        console.log(`⚠️ Admin validation failed for ID: ${studentId}`);
        return res.status(401).json({
          success: false,
          error: "Invalid administrator credentials. Access denied.",
          message: "Invalid administrator credentials. Access denied."
        });
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
    if (profile && profile.password === password) {
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
    const { studentId, email, course, yearLevel, password } = req.body;

    if (!studentId || !email || !course || !yearLevel || !password) {
      console.log("⚠️  /api/auth/signup — Missing registration details.");
      return res.status(400).json({
        success: false,
        error: "All registration fields (ID, email, course, year level, password) are required.",
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

    // Execute a clean record insert call directly to our Supabase 'profiles' data table
    const { error: insertError } = await supabase
      .from("profiles")
      .insert([{ 
        student_id: studentId, 
        email: email, 
        course: course, 
        year_level: yearLevel, 
        password: password 
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
//  POST /api/tickets/submit — Submit a user-confirmed support ticket (Max 3 Open)
// ============================================================================
app.post("/api/tickets/submit", async (req, res) => {
  try {
    const { studentId, unresolvedInquiry } = req.body;

    if (!studentId || !unresolvedInquiry) {
      return res.status(400).json({
        success: false,
        message: "Missing studentId or unresolvedInquiry in request body."
      });
    }

    console.log(`🎫 Submission request for student: ${studentId}`);

    // Step A (Check Limit)
    const { data: existingTickets, error: countError } = await supabase
      .from('tickets')
      .select('id')
      .eq('student_id', studentId)
      .eq('status', 'Pending');

    if (countError) {
      console.error("❌ Error counting existing tickets:", countError.message);
      return res.status(500).json({ success: false, message: "Database query failed.", details: countError.message });
    }

    // Step B (Enforce Rule)
    if (existingTickets && existingTickets.length >= 3) {
      return res.status(400).json({ 
        success: false, 
        message: "Submission Blocked: You currently have 3 pending support tickets. Please wait for the ICCT Administration to resolve your open tickets before submitting a new one." 
      });
    }

    // Step C (Insert Ticket)
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
    console.error("❌ /api/tickets/submit error:", err);
    return res.status(500).json({
      success: false,
      message: "Internal server error.",
      details: err.message
    });
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

    // 2. AGGREGATE CATEGORY DISTRIBUTION
    const { data: chatLogs, error: logsError } = await supabase
      .from("chat_logs")
      .select("message_text")
      .eq("sender", "user");

    if (logsError) {
      console.error("❌ Stats error fetching chat logs:", logsError.message);
      return res.status(500).json({ error: logsError.message });
    }

    const categories = {
      Enrollment: 0,
      Tuition: 0,
      "Document Requests": 0,
      Scheduling: 0
    };

    if (chatLogs) {
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
    }

    // 3. AI RESOLUTION MATH
    const totalInquiries = chatLogs?.length ?? 0;
    const selfServed = Math.max(0, totalInquiries - totalTickets);
    const resolutionRateVal = totalInquiries > 0 ? (selfServed / totalInquiries) * 100 : 100.0;
    const resolutionRate = `${resolutionRateVal.toFixed(1)}%`;

    return res.status(200).json({
      totalInquiries,
      pendingTickets,
      resolutionRate,
      categories,
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
    const { data, error } = await supabase
      .from("tickets")
      .select("*")
      .order("created_at", { ascending: false });

    if (error) {
      console.error("❌ Error fetching global tickets:", error.message);
      return res.status(500).json({ error: error.message });
    }

    return res.json(data);
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

console.log("\n🚀 UniBot Backend — ICCT Colleges Capstone");
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
