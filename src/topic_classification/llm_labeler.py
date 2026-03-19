import os
import json
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

SENTENCES_PATH = Path("data/corpus/sentences.parquet")
OUTPUT_PATH = Path("data/labels/llm_prelabels.parquet")

TOPICS = {
    "E": "Environmental - khí thải, năng lượng, môi trường, tín dụng xanh",
    "S_labor": "Social (Labor) - nhân viên, đào tạo, phúc lợi, an toàn lao động",
    "S_community": "Social (Community) - cộng đồng, từ thiện, CSR, trách nhiệm xã hội",
    "S_product": "Social (Product) - khách hàng, bảo mật, chất lượng dịch vụ",
    "G": "Governance - quản trị rủi ro, minh bạch, tuân thủ, kiểm soát nội bộ",
    "Non_ESG": "Non-ESG - không liên quan ESG, thông tin tài chính thuần túy, tiểu sử",
}

SYSTEM_PROMPT = """Bạn là chuyên gia phân loại ESG cho báo cáo thường niên ngân hàng Việt Nam.

Phân loại câu sau vào MỘT trong 6 nhãn:
- E: Environmental (môi trường, khí thải, năng lượng xanh, tín dụng xanh)
- S_labor: Social-Labor (nhân viên, đào tạo, phúc lợi, an toàn lao động) - KHÔNG bao gồm tiểu sử lãnh đạo
- S_community: Social-Community (cộng đồng, từ thiện, CSR)
- S_product: Social-Product (khách hàng, bảo mật, chất lượng dịch vụ)
- G: Governance (quản trị rủi ro, minh bạch, tuân thủ, kiểm soát nội bộ) - KHÔNG bao gồm tiểu sử HĐQT
- Non_ESG: Không phải ESG (tài chính thuần túy, tiểu sử cá nhân, thông tin chung)

Lưu ý quan trọng:
- Tiểu sử lãnh đạo/HĐQT → Non_ESG
- "Quản trị" trong ngữ cảnh corporate governance thật sự → G
- "Quản trị" trong tiểu sử (thành viên HĐQT) → Non_ESG

Trả lời ĐÚNG JSON format: {"topic": "<label>", "confidence": <0.0-1.0>, "reason": "<brief>"}"""


def create_prompt(row: pd.Series) -> str:
    context = ""
    if row["ctx_prev"]:
        context += f"[Câu trước]: {row['ctx_prev']}\n"
    context += f"[Câu cần phân loại]: {row['sentence']}\n"
    if row["ctx_next"]:
        context += f"[Câu sau]: {row['ctx_next']}\n"
    context += f"[Section]: {row['section_title']}"
    return context


def call_gemini(prompt: str, client, model_name: str = "gemini-3.1-flash-lite-preview") -> dict:
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[
                {"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\n" + prompt}]}
            ],
            config={
                "temperature": 0.1,
                "max_output_tokens": 200,
            }
        )
        text = response.text.strip()
        if "{" in text and "}" in text:
            json_str = text[text.find("{"):text.rfind("}")+1]
            result = json.loads(json_str)
            if "topic" in result:
                return result
        return {"topic": "Non_ESG", "confidence": 0.5, "reason": "parse_error"}
    except Exception as e:
        return {"topic": "Non_ESG", "confidence": 0.5, "reason": f"error: {str(e)[:50]}"}


def process_row(args):
    idx, row, client = args
    prompt = create_prompt(row)
    result = call_gemini(prompt, client)
    return {
        "sent_id": row["sent_id"],
        "sentence": row["sentence"],
        "ctx_prev": row["ctx_prev"],
        "ctx_next": row["ctx_next"],
        "section_title": row["section_title"],
        "block_type": row["block_type"],
        "doc_id": row["doc_id"],
        "bank": row["bank"],
        "year": row["year"],
        "llm_topic": result.get("topic", "Non_ESG"),
        "llm_conf": result.get("confidence", 0.5),
        "llm_reason": result.get("reason", ""),
    }


def run_labeling(target_per_class: int = 500, workers: int = 15):
    from google import genai
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    
    print(f"Loading corpus and weak labels to target {target_per_class} per class...")
    sents = pd.read_parquet(SENTENCES_PATH)
    weak_path = Path("data/labels/weak_labels.parquet")
    if weak_path.exists():
        weak = pd.read_parquet(weak_path)
        df = sents.merge(weak[["sent_id", "weak_topic", "weak_conf"]], on="sent_id", how="left")
    else:
        df = sents.copy()
        df["weak_topic"] = None
        df["weak_conf"] = 0.0

    # Filter out obvious noise
    df = df[~df["sentence"].str.contains(r"<!-- image -->|^\d+$", regex=True, na=False)]
    df = df[df["sentence"].str.len() >= 20]
    
    results = []
    labeled_ids = set()
    counts = {t: 0 for t in TOPICS.keys()}
    
    iteration = 1
    while True:
        needed_classes = [t for t, c in counts.items() if c < target_per_class]
        if not needed_classes:
            print(f"\nTarget of {target_per_class} reached for all classes!")
            break
            
        print(f"\n--- Iteration {iteration} ---")
        print("Current counts:", counts)
        print("Needed classes:", needed_classes)
        
        candidates = df[~df["sent_id"].isin(labeled_ids)]
        batch_samples = []
        
        for t in needed_classes:
            needed_amount = target_per_class - counts[t]
            # Try to get candidates matching weak_topic first
            subset = candidates[(candidates["weak_topic"] == t) & (candidates["weak_conf"] >= 0.3)]
            n_sample = min(len(subset), max(50, needed_amount * 2))  # over-sample a bit because LLM might change its mind
            
            if n_sample > 0:
                batch_samples.append(subset.sample(n=n_sample, random_state=iteration))
            else:
                # Fallback to unlabeled sentences
                fallback = candidates[candidates["weak_topic"].isna()]
                n_fallback = min(len(fallback), max(50, needed_amount))
                if n_fallback > 0:
                    batch_samples.append(fallback.sample(n=n_fallback, random_state=iteration))
                    
        if not batch_samples:
            print("No more candidates available!")
            break
            
        batch_df = pd.concat(batch_samples).drop_duplicates(subset=["sent_id"])
        
        # If batch gets too large somehow, cap it
        if len(batch_df) > 2000:
            batch_df = batch_df.sample(2000, random_state=iteration)
            
        print(f"Sampling {len(batch_df)} sentences for this batch...")
        
        args_list = [(idx, row, client) for idx, row in batch_df.iterrows()]
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_row, args): args[0] for args in args_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Labeling"):
                res = future.result()
                results.append(res)
                labeled_ids.add(res["sent_id"])
                
                topic = res.get("llm_topic", "Non_ESG")
                if topic in counts:
                    counts[topic] += 1
                    
        # Save intermediate checkpoint
        pd.DataFrame(results).to_parquet(OUTPUT_PATH, index=False)
        iteration += 1
        
    df_results = pd.DataFrame(results)
    print(f"\nFinal LLM labels distribution ({len(df_results)} total):")
    print(df_results["llm_topic"].value_counts())
    print(f"Saved: {OUTPUT_PATH}")
    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=500, help="Target samples per class")
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()
    
    run_labeling(args.target, workers=args.workers)

