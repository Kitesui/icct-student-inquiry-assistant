// ============================================================================
//  import-vector-data.js — One-Time Data Migration Script
// ============================================================================
//  Reads school_knowledge.csv, generates Gemini embeddings for each row,
//  and inserts them into the Supabase 'school_knowledge' table.
//
//  Usage:  node import-vector-data.js
// ============================================================================

import { createRequire } from "module";
const require = createRequire(import.meta.url);
require('dotenv').config();

import fs from "fs";
import path from "path";
import { GoogleGenAI } from "@google/genai";
import { createClient } from "@supabase/supabase-js";

// ── Initialise Clients (reuses the same .env keys as server.js) ─────────────
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !supabaseServiceKey) {
    console.error("Missing cloud connection keys inside process.env!");
    process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseServiceKey);

// ── Embedding model name ────────────────────────────────────────────────────
const EMBEDDING_MODEL = "gemini-embedding-001";

// ── CSV file path ───────────────────────────────────────────────────────────
const CSV_PATH = path.resolve("data/school_knowledge.csv");

// ============================================================================
//  CSV Parser — handles quoted fields that may contain commas & newlines
// ============================================================================

/**
 * parseCSVLine — Splits a single CSV line into field values using a
 * state-machine approach to correctly handle quoted fields with commas.
 */
function parseCSVLine(line) {
  const fields = [];
  let current = "";
  let insideQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];

    if (char === '"') {
      if (insideQuotes && i + 1 < line.length && line[i + 1] === '"') {
        current += '"';
        i++; // skip escaped quote
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

  fields.push(current.trim());
  return fields;
}

/**
 * parseSchoolKnowledgeCSV — Reads the CSV and returns an array of
 * { category, title, content } objects.
 */
function parseSchoolKnowledgeCSV(filePath) {
  const rawText = fs.readFileSync(filePath, "utf-8");
  const rawLines = rawText.split("\n");
  const entries = [];

  // ── Reassemble multi-line quoted fields ────────────────────────────────
  const logicalLines = [];
  let buffer = "";
  let openQuotes = false;

  for (const raw of rawLines) {
    const line = raw.replace(/\r$/, "");

    if (!openQuotes) {
      buffer = line;
    } else {
      buffer += "\n" + line;
    }

    const quoteCount = (buffer.match(/"/g) || []).length;
    openQuotes = quoteCount % 2 !== 0;

    if (!openQuotes) {
      logicalLines.push(buffer);
      buffer = "";
    }
  }
  if (buffer) logicalLines.push(buffer);

  // ── Parse each logical line ────────────────────────────────────────────
  for (const line of logicalLines) {
    const fields = parseCSVLine(line);

    const category = (fields[0] || "").trim();
    const title    = (fields[1] || "").trim();
    const content  = (fields[2] || "").trim();

    // Skip the header row and any blank rows
    if (!category || category.toLowerCase() === "category") continue;
    if (!content) continue;

    entries.push({ category, title, content });
  }

  return entries;
}

// ── Helpers ──────────────────────────────────────────────────────────────────
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * embedWithRetry — Calls the Gemini embedding API with exponential backoff
 * on 429 (rate-limit) errors.  Retries up to `maxRetries` times.
 */
async function embedWithRetry(content, maxRetries = 4) {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const result = await ai.models.embedContent({
        model: EMBEDDING_MODEL,
        contents: content,
        config: { outputDimensionality: 1536 },
      });
      return result.embeddings[0].values;
    } catch (err) {
      const is429 = err.message?.includes("429") || err.message?.includes("RESOURCE_EXHAUSTED");
      if (is429 && attempt < maxRetries) {
        const backoff = Math.pow(2, attempt) * 1000; // 1s, 2s, 4s, 8s
        console.log(`  ⏳ Rate limited — retrying in ${backoff / 1000}s (attempt ${attempt + 1}/${maxRetries})…`);
        await sleep(backoff);
      } else {
        throw err; // non-retryable or exhausted retries
      }
    }
  }
}

// ============================================================================
//  Main Migration
// ============================================================================

async function main() {
  console.log("══════════════════════════════════════════════════════════");
  console.log("  📥  ICCT UniBot — Vector Data Import Script");
  console.log("══════════════════════════════════════════════════════════\n");

  // ── Step 1: Read & parse the CSV ──────────────────────────────────────
  console.log(`Reading CSV: ${CSV_PATH}\n`);
  const rows = parseSchoolKnowledgeCSV(CSV_PATH);
  console.log(`Found ${rows.length} data rows to process.\n`);

  if (rows.length === 0) {
    console.log("⚠️  No rows found. Exiting.");
    return;
  }

  // ── Step 2: Clear existing rows for a clean import ────────────────────
  console.log("🗑️  Clearing existing rows from school_knowledge table…");
  const { error: deleteError } = await supabase
    .from("school_knowledge")
    .delete()
    .gte("id", 0); // matches all rows

  if (deleteError) {
    console.error(`❌ Failed to clear table: ${deleteError.message}`);
    console.log("Continuing anyway — duplicates may occur.\n");
  } else {
    console.log("✅ Table cleared.\n");
  }

  // ── Step 3: Loop through each row — embed & insert ────────────────────
  let successCount = 0;
  let errorCount = 0;

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];

    try {
      // Generate embedding with automatic retry on rate limits
      const embeddingVector = await embedWithRetry(row.content);

      // Insert into Supabase 'school_knowledge' table
      const { error: insertError } = await supabase
        .from("school_knowledge")
        .insert({
          category: row.category,
          title: row.title,
          content: row.content,
          embedding: embeddingVector,
        });

      if (insertError) {
        console.error(`  ❌ Row ${i + 1} INSERT failed (${row.title}): ${insertError.message}`);
        errorCount++;
        continue;
      }

      successCount++;
      console.log(`  ✅ Processed row: ${row.title}  [${i + 1}/${rows.length}]`);

    } catch (err) {
      console.error(`  ❌ Row ${i + 1} ERROR (${row.title}): ${err.message}`);
      errorCount++;
    }

    // Pace requests to stay under Gemini free-tier rate limits
    if (i < rows.length - 1) await sleep(1500);
  }

  // ── Summary ───────────────────────────────────────────────────────────
  console.log("\n══════════════════════════════════════════════════════════");
  console.log(`  ✅ Success: ${successCount}  |  ❌ Failed: ${errorCount}  |  Total: ${rows.length}`);
  console.log(`  🎉 Successfully wrote a total of ${successCount} rows to the cloud database.`);
  console.log("══════════════════════════════════════════════════════════\n");
}

// ── Execute ─────────────────────────────────────────────────────────────────
main().catch((err) => {
  console.error("\n🔥 Fatal error during import:", err);
  process.exit(1);
});
