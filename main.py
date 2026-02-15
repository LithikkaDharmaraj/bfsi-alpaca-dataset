from dataset_service import match_dataset  # Tier 1
from rag_service import retrieve_context  # Tier 2
from slm_service import generate_response  # Tier 3
from prompt_builder import build_rag_prompt, build_prompt

DATASET_THRESHOLD = 0.80
RAG_THRESHOLD = 0.65

def main():
    user_query = input("\nEnter customer query: ")

    # Tier 1 - Dataset Match
    dataset_answer, dataset_score = match_dataset(user_query)

    print(f"\n[Dataset Score]: {dataset_score}")

    if dataset_score >= DATASET_THRESHOLD:
        print("[Layer Used]: Tier 1 - DATASET")
        print("\nResponse:\n")
        print(dataset_answer)
        return

    # Tier 2 - RAG Retrieval
    context, rag_score = retrieve_context(user_query)

    print(f"\n[RAG Score]: {rag_score}")

    if context and rag_score >= RAG_THRESHOLD:
        print("[Layer Used]: Tier 2 - RAG")

        prompt = build_rag_prompt(user_query, context)
        response = generate_response(prompt)

        print("\nResponse:\n")
        print(response)
        return

    # Tier 3 - SLM Fallback
    print("[Layer Used]: Tier 3 - SLM FALLBACK")

    prompt = build_prompt(user_query)
    response = generate_response(prompt)

    print("\nResponse:\n")
    print(response)

if __name__ == "__main__":
    main()