"""Financial Optimization Orchestrator Agent.

This agent demonstrates the orchestrator-workers pattern using Claude Agent SDK.
It fetches financial data from MCP servers and coordinates specialized sub-agents
to provide comprehensive financial optimization recommendations.
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AgentDefinition,
    AssistantMessage,
    ResultMessage,
    PermissionResultAllow,
)
from claude_agent_sdk.types import TextBlock


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


DATA_DIR: Path = Path(__file__).parent.parent / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw_data"
AGENT_OUTPUTS_DIR: Path = DATA_DIR / "agent_outputs"
PROMPTS_DIR: Path = Path(__file__).parent / "prompts"


def _ensure_directories():
    """Ensure all required directories exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    AGENT_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_json(
    data: dict,
    filename: str
):
    """Save data to JSON file."""
    filepath = RAW_DATA_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved data to {filepath}")


def _load_prompt(filename: str) -> str:
    """Load prompt from prompts directory.

    Args:
        filename: Name of the prompt file in the prompts directory

    Returns:
        Prompt text content
    """
    prompt_path = PROMPTS_DIR / filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text()


async def _auto_approve_all(
    tool_name: str,
    input_data: dict,
    context
) -> PermissionResultAllow:
    """Auto-approve all tools without prompting.

    Args:
        tool_name: Name of the tool being requested
        input_data: Input parameters for the tool
        context: Additional request context

    Returns:
        PermissionResultAllow to allow the tool call
    """
    logger.debug(f"Auto-approving tool: {tool_name}")
    return PermissionResultAllow()


def _detect_subscriptions(
    bank_transactions: list[dict],
    credit_card_transactions: list[dict]
) -> list[dict]:
    """Detect subscription services from recurring transactions.

    Filters all transactions marked as recurring=True and extracts
    subscription name, amount, and frequency.

    Args:
        bank_transactions: List of bank transaction dicts
        credit_card_transactions: List of credit card transaction dicts

    Returns:
        List of subscription dictionaries with service name, amount, frequency
    """
    subscriptions = []

    all_transactions = bank_transactions + credit_card_transactions

    for transaction in all_transactions:
        if transaction.get("recurring") is True:
            amount = transaction.get("amount", 0)
            # Subscriptions are outflows (negative amounts)
            if amount < 0:
                subscriptions.append({
                    "service": transaction.get("description", "Unknown"),
                    "amount": abs(amount),
                    "frequency": "monthly",
                    "category": transaction.get("category", "Unknown")
                })

    logger.info(f"Detected {len(subscriptions)} subscriptions from {len(all_transactions)} transactions")
    return subscriptions


async def _fetch_financial_data(
    username: str,
    start_date: str,
    end_date: str
) -> tuple[dict, dict]:
    """Fetch data from Bank and Credit Card MCP servers.

    Configures MCP server connections (ports 5001 and 5002),
    calls get_bank_transactions and get_credit_card_transactions tools,
    and saves raw data to files.

    Args:
        username: Username for the account
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Tuple of (bank_data, credit_card_data) dictionaries
    """
    logger.info(f"Fetching financial data for {username} from {start_date} to {end_date}")

    mcp_servers = {
        "Bank Account Server": {
            "type": "http",
            "url": "http://127.0.0.1:5001/mcp"
        },
        "Credit Card Server": {
            "type": "http",
            "url": "http://127.0.0.1:5002/mcp"
        }
    }

    fetch_prompt = f"""Fetch financial data for user '{username}' from {start_date} to {end_date}.

Please do the following IN ORDER:
1. Call the get_bank_transactions tool with username="{username}", start_date="{start_date}", end_date="{end_date}"
2. Call the get_credit_card_transactions tool with username="{username}", start_date="{start_date}", end_date="{end_date}"
3. Return ONLY a JSON object with exactly this structure (no markdown, no explanation):
{{
  "bank_data": <full bank tool result as JSON>,
  "credit_card_data": <full credit card tool result as JSON>
}}
"""

    options = ClaudeAgentOptions(
        model="haiku",
        system_prompt="You are a data fetching assistant. Call the specified MCP tools and return the results as a JSON object.",
        mcp_servers=mcp_servers,
        can_use_tool=_auto_approve_all,
    )

    bank_data = {}
    credit_card_data = {}

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(fetch_prompt)

            full_text = ""
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            full_text += block.text
                elif isinstance(message, ResultMessage):
                    logger.info(f"Data fetch complete. Duration: {message.duration_ms}ms")
                    break

            # Try to parse JSON from the response
            if full_text.strip():
                try:
                    # Strip markdown code fences if present
                    text = full_text.strip()
                    if text.startswith("```"):
                        lines = text.split("\n")
                        text = "\n".join(lines[1:-1])
                    result = json.loads(text)
                    bank_data = result.get("bank_data", {})
                    credit_card_data = result.get("credit_card_data", {})
                    logger.info(f"Parsed bank transactions: {len(bank_data.get('transactions', []))}")
                    logger.info(f"Parsed credit card transactions: {len(credit_card_data.get('transactions', []))}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse JSON response: {e}. Response was: {full_text[:500]}")

    except Exception as e:
        logger.error(f"Error fetching financial data: {e}", exc_info=True)
        logger.error("Make sure MCP servers are running:")
        logger.error("  cd mcp_servers && uv run python bank_server.py")
        logger.error("  cd mcp_servers && uv run python credit_card_server.py")
        raise

    # Save raw data
    _save_json(bank_data, "bank_transactions.json")
    _save_json(credit_card_data, "credit_card_transactions.json")

    return bank_data, credit_card_data


async def _run_orchestrator(
    username: str,
    start_date: str,
    end_date: str,
    user_query: str
):
    """Main orchestrator agent logic.

    Implements the orchestrator pattern:
    1. Fetch data from MCP servers
    2. Perform initial analysis (detect subscriptions)
    3. Define sub-agents (research, negotiation, tax)
    4. Configure orchestrator with sub-agents and MCP servers
    5. Execute orchestrator with Claude Agent SDK

    Args:
        username: Username for the account
        start_date: Start date for analysis
        end_date: End date for analysis
        user_query: User's financial question/request
    """
    logger.info("Starting financial optimization orchestrator")
    logger.info(f"User query: {user_query}")

    _ensure_directories()

    # Step 1: Fetch financial data from MCP servers
    bank_data, credit_card_data = await _fetch_financial_data(
        username,
        start_date,
        end_date
    )

    # Step 2: Initial analysis
    logger.info("Performing initial analysis...")

    bank_transactions = bank_data.get("transactions", [])
    credit_card_transactions = credit_card_data.get("transactions", [])

    subscriptions = _detect_subscriptions(
        bank_transactions,
        credit_card_transactions
    )

    logger.info(f"Detected {len(subscriptions)} subscriptions")

    # Step 3: Define sub-agents
    research_agent = AgentDefinition(
        description="Research cheaper alternatives for subscriptions and services",
        prompt=_load_prompt("research_agent_prompt.txt"),
        tools=["write"],
        model="haiku"
    )

    negotiation_agent = AgentDefinition(
        description="Create negotiation strategies and scripts for bills and services",
        prompt=_load_prompt("negotiation_agent_prompt.txt"),
        tools=["write"],
        model="haiku"
    )

    tax_agent = AgentDefinition(
        description="Identify tax-deductible expenses and optimization opportunities",
        prompt=_load_prompt("tax_agent_prompt.txt"),
        tools=["write"],
        model="haiku"
    )

    agents = {
        "research_agent": research_agent,
        "negotiation_agent": negotiation_agent,
        "tax_agent": tax_agent,
    }

    # Step 4: Configure orchestrator agent with sub-agents and MCP servers
    mcp_servers = {
        "Bank Account Server": {
            "type": "http",
            "url": "http://127.0.0.1:5001/mcp"
        },
        "Credit Card Server": {
            "type": "http",
            "url": "http://127.0.0.1:5002/mcp"
        }
    }

    working_dir = Path(__file__).parent.parent  # personal-financial-analyst/

    options = ClaudeAgentOptions(
        model="sonnet",
        system_prompt=_load_prompt("orchestrator_system_prompt.txt"),
        mcp_servers=mcp_servers,
        agents=agents,
        can_use_tool=_auto_approve_all,
        cwd=str(working_dir)
    )

    # Step 5: Run orchestrator with Claude Agent SDK
    prompt = f"""Analyze my financial data and {user_query}

I have:
- {len(bank_transactions)} bank transactions
- {len(credit_card_transactions)} credit card transactions
- {len(subscriptions)} identified subscriptions

Subscriptions detected:
{json.dumps(subscriptions, indent=2)}

Please:
1. Identify opportunities for savings
2. Delegate research to the research agent
3. Delegate negotiation strategies to the negotiation agent
4. Delegate tax analysis to the tax agent
5. Read their results and create a final report at data/final_report.md
"""

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(block.text, end='', flush=True)
                elif isinstance(message, ResultMessage):
                    logger.info(f"Duration: {message.duration_ms}ms")
                    logger.info(f"Cost: ${message.total_cost_usd:.4f}")
                    break

    except Exception as e:
        logger.error(f"Error during orchestration: {e}", exc_info=True)
        logger.error("\nTroubleshooting:")
        logger.error("1. Make sure MCP servers are running")
        logger.error("   curl http://127.0.0.1:5001/health")
        logger.error("   curl http://127.0.0.1:5002/health")
        logger.error("2. Check that ANTHROPIC_API_KEY is set")
        logger.error("3. If running inside Claude Code: unset CLAUDECODE")
        raise

    # Step 6: Complete
    logger.info("Orchestration complete. Check data/final_report.md for results.")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Financial Optimization Orchestrator Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Basic analysis
    uv run python financial_orchestrator.py \\
        --username john_doe \\
        --start-date 2026-01-01 \\
        --end-date 2026-01-31 \\
        --query "How can I save $500 per month?"

    # Subscription analysis
    uv run python financial_orchestrator.py \\
        --username jane_smith \\
        --start-date 2026-01-01 \\
        --end-date 2026-01-31 \\
        --query "Analyze my subscriptions and find better deals"
"""
    )

    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="Username for account (john_doe or jane_smith)"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User's financial question or request"
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = _parse_args()

    await _run_orchestrator(
        username=args.username,
        start_date=args.start_date,
        end_date=args.end_date,
        user_query=args.query
    )


if __name__ == "__main__":
    asyncio.run(main())
