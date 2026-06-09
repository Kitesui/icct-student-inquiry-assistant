import "dotenv/config";
import { GoogleGenAI } from "@google/genai";
import { createClient } from "@supabase/supabase-js";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

const testQuery = "PAANO KUMUHA NG PROSPECTUS?";

console.log(`\nTest query: "${testQuery}"\n`);

// Step 1: Embed
const emb = await ai.models.embedContent({
  model: "gemini-embedding-001",
  contents: testQuery,
  config: { outputDimensionality: 1536 },
});
const vec = emb.embeddings[0].values;
console.log("Vector length:", vec.length);

// Step 2: RPC call
const { data, error } = await supabase.rpc("match_knowledge", {
  query_embedding: vec,
  match_threshold: 0.35,
  match_count: 3,
});

if (error) {
  console.log("RPC Error:", error.message);
} else {
  console.log("Matches found:", data.length);
  data.forEach((r) => {
    console.log(`  ${r.similarity.toFixed(4)} — ${r.title}`);
  });
}
