import sys
from orchestrator import MedscapeOrchestrator
from clients import MockLLMClient, OpenAIClient


def demo_mode():
    print("=" * 70)
    print("  MEDSCAPE AGENTIC AI SYSTEM - DEMO MODE")
    print("  Architecture: LLM Tool Orchestration")
    print("=" * 70)
    
    orchestrator = MedscapeOrchestrator(llm=MockLLMClient())
    
    queries = [
        "Compare tactic performance for Cardiology vs Oncology in 2025Q2 and recommend where to shift 20% spend.",
        "For Endocrinology, which tactics have the highest ROI stability?",
        "Should we emphasize Webinars or Email for Oncology? Check the knowledge base."
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n\n{'#' * 70}")
        print(f"  QUERY {i}")
        print(f"{'#' * 70}")
        
        result = orchestrator.process(query, verbose=True)
        
        print(f"\n{'─' * 60}")
        print("📊 FINAL OUTPUT:")
        print(f"{'─' * 60}")
        print(result)
        
        if i < len(queries):
            input("\nPress Enter for next query...")
    
    print("\n" + "=" * 70)
    print("  DEMO COMPLETE!")
    print("=" * 70)


def interactive_mode():
    """Interactive mode for custom questions."""
    print("=" * 70)
    print("  MEDSCAPE AGENTIC AI SYSTEM - INTERACTIVE MODE")
    print("=" * 70)
    print("\nType your campaign performance questions below.")
    print("Available data: Cardiology, Oncology, Endocrinology | 2025Q1-Q3")
    print("Available tactics: Email, Display, Webinar, HCP_Newsletter, Social")
    print("\nCommands: 'help' for examples, 'exit' to quit")
    print("-" * 70)
    
    orchestrator = MedscapeOrchestrator(llm=MockLLMClient())
    
    example_queries = [
        "Compare Cardiology vs Oncology in 2025Q2",
        "Which tactics have the highest ROI stability for Endocrinology?",
        "Should we use Webinars or Email for Oncology?",
        "What's the best tactic for Cardiology?",
        "Recommend where to shift spend for Oncology"
    ]
    
    while True:
        try:
            print()
            user_input = input("🔍 Your question: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye! 👋")
                break
            
            if user_input.lower() == 'help':
                print("\n📝 Example queries:")
                for q in example_queries:
                    print(f"   • {q}")
                continue
            
            result = orchestrator.process(user_input, verbose=True)
            
            print(f"\n{'─' * 60}")
            print("📊 FINAL OUTPUT:")
            print(f"{'─' * 60}")
            print(result)
            print(f"{'─' * 60}")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try a different question.")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode in ['--demo', '-d', 'demo']:
            demo_mode()
        elif mode in ['--interactive', '-i', 'interactive']:
            interactive_mode()
    else:
        # Show menu
        print("=" * 70)
        print("  MEDSCAPE AGENTIC AI SYSTEM")
        print("=" * 70)
        print("\nSelect mode:")
        print("  1. Demo (run sample queries)")
        print("  2. Interactive (type your own questions)")
        print("  3. Exit")
        print()
        
        choice = input("Enter choice (1/2/3): ").strip()
        
        if choice == '1':
            demo_mode()
        elif choice == '2':
            interactive_mode()
        elif choice == '3':
            print("Goodbye! 👋")
        else:
            print("Invalid choice. Running demo mode...")
            demo_mode()


if __name__ == "__main__":
    main()
