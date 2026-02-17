import os
import json
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

ESG_SENTENCES_PATH = Path("data/corpus/esg_sentences.parquet")
OUTPUT_PATH = Path("data/labels/action_llm.parquet")

LABELS = ["Implemented", "Planning", "Indeterminate"]

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích ESG cho báo cáo thường niên ngân hàng Việt Nam.

Phân loại câu sau vào MỘT trong 3 nhãn HÀNH ĐỘNG:

1. **Implemented** - Đã triển khai / có kết quả cụ thể:
   - Có động từ quá khứ: "đã triển khai", "đã thực hiện", "đã hoàn thành"
   - Có KPI / số liệu cụ thể: %, tỷ, triệu, tấn CO2, kWh
   - Có mốc thời gian đã qua: "năm 2023", "trong năm qua"
   - Ví dụ: "Năm 2023, Ngân hàng đã giảm 15% lượng điện tiêu thụ."

2. **Planning** - Kế hoạch / mục tiêu tương lai:
   - Có động từ tương lai: "sẽ", "dự kiến", "kế hoạch", "mục tiêu"
   - Có mốc thời gian tương lai: "đến 2025", "năm 2030"
   - Ví dụ: "Ngân hàng đặt mục tiêu trung hòa carbon vào năm 2050."

3. **Indeterminate** - Cam kết chung chung, không rõ hành động:
   - Ngôn ngữ mơ hồ: "cam kết", "hướng tới", "tăng cường", "đẩy mạnh"
   - KHÔNG có số liệu cụ thể
   - KHÔNG có mốc thời gian rõ ràng
   - Ví dụ: "Ngân hàng cam kết phát triển bền vững và bảo vệ môi trường."

Trả lời ĐÚNG JSON format: {"action": "<Implemented|Planning|Indeterminate>", "confidence": <0.0-1.0>, "reason": "<brief>"}"""


def create_prompt(row: pd.Series) -> str:
    context = ""
    if row["ctx_prev"]:
        context += f"[Câu trước]: {row['ctx_prev']}\n"
    context += f"[Câu cần phân loại]: {row['sentence']}\n"
    if row["ctx_next"]:
        context += f"[Câu sau]: {row['ctx_next']}\n"
    context += f"[ESG Topic]: {row.get('predicted_label', 'Unknown')}"
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
            if "action" in result:
                return result
        return {"action": "Indeterminate", "confidence": 0.5, "reason": "parse_error"}
    except Exception as e:
        return {"action": "Indeterminate", "confidence": 0.5, "reason": f"error: {str(e)[:50]}"}


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
        "predicted_label": row.get("predicted_label", ""),
        "llm_action": result.get("action", "Indeterminate"),
        "llm_conf": result.get("confidence", 0.5),
        "llm_reason": result.get("reason", ""),
    }


def run_labeling(target_per_class: int = 500, workers: int = 15):
    from google import genai
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    
    print(f"Loading ESG sentences and weak labels to target {target_per_class} per class...")
    sents = pd.read_parquet(ESG_SENTENCES_PATH)
    weak_path = Path("data/labels/action_weak.parquet")
    
    if weak_path.exists():
        weak = pd.read_parquet(weak_path)
        df = sents.merge(weak[["sent_id", "weak_action", "weak_conf"]], on="sent_id", how="left")
    else:
        df = sents.copy()
        df["weak_action"] = None
        df["weak_conf"] = 0.0

    # Filter out obvious noise
    df = df[~df["sentence"].str.contains(r"<!-- image -->|^\d+$", regex=True, na=False)]
    df = df[df["sentence"].str.len() >= 20]
    
    results = []
    labeled_ids = set()
    counts = {t: 0 for t in LABELS}
    
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
            # Try to get candidates matching weak_action first
            subset = candidates[(candidates["weak_action"] == t) & (candidates["weak_conf"] >= 0.4)]
            # Over-sample highly to ensure we hit the target quickly
            n_sample = min(len(subset), max(50, needed_amount * 3))
            
            if n_sample > 0:
                batch_samples.append(subset.sample(n=n_sample, random_state=iteration))
            else:
                # Fallback to unlabeled sentences
                fallback = candidates[candidates["weak_action"].isna()]
                n_fallback = min(len(fallback), max(50, needed_amount * 3))
                if n_fallback > 0:
                    batch_samples.append(fallback.sample(n=n_fallback, random_state=iteration))
                    
        if not batch_samples:
            print("No more candidates available!")
            break
            
        batch_df = pd.concat(batch_samples).drop_duplicates(subset=["sent_id"])
        
        if len(batch_df) > 3000:
            batch_df = batch_df.sample(3000, random_state=iteration)
            
        print(f"Sampling {len(batch_df)} sentences for this batch...")
        
        args_list = [(idx, row, client) for idx, row in batch_df.iterrows()]
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_row, args): args[0] for args in args_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Labeling"):
                res = future.result()
                results.append(res)
                labeled_ids.add(res["sent_id"])
                
                action = res.get("llm_action", "Indeterminate")
                if action in counts:
                    counts[action] += 1
                    
        # Save intermediate checkpoint
        pd.DataFrame(results).to_parquet(OUTPUT_PATH, index=False)
        iteration += 1
        
    df_results = pd.DataFrame(results)
    print(f"\nFinal LLM labels distribution ({len(df_results)} total):")
    print(df_results["llm_action"].value_counts())
    print(f"Saved: {OUTPUT_PATH}")
    return df_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=500, help="Target samples per class")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--test", action="store_true", help="Test API connection")
    args = parser.parse_args()
    
    if args.test:
        test_api()
    else:
        run_labeling(args.target, workers=args.workers)
