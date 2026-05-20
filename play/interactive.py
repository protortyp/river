#!/usr/bin/env python3
"""
Interactive heads-up poker game against a trained agent.

Play Texas Hold'em poker against agents from the league pool.
Features a rich terminal UI with Unicode cards and live game state display.

Usage:
    uv run python play/interactive.py
    uv run python play/interactive.py --pool-dir checkpoints/league_pool --seat 0
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import warp as wp
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

# Add project root to sys.path
# This allows imports like 'from train.league import ...' to work when running
# this script directly.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from train.league import UnifiedOpponentPool

from gpu_poker import constants as c
from gpu_poker.env import WarpPokerEnv
from gpu_poker.policy import PokerPolicyNet, raise_frac_to_amount

# ============================================================================
# Constants
# ============================================================================

RANK_NAMES = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUIT_SYMBOLS = ["♣", "♦", "♥", "♠"]
SUIT_COLORS = ["white", "blue", "red", "white"]  # Club, Diamond, Heart, Spade

STAGE_NAMES = {
    c.STAGE_PREFLOP: "Preflop",
    c.STAGE_FLOP: "Flop",
    c.STAGE_TURN: "Turn",
    c.STAGE_RIVER: "River",
    c.STAGE_SHOWDOWN: "Showdown",
    c.STAGE_TERMINAL: "Terminal",
}

ACTION_NAMES = {
    c.ACTION_FOLD: "Fold",
    c.ACTION_CHECK: "Check",
    c.ACTION_CALL: "Call",
    c.ACTION_RAISE: "Raise",
}

console = Console()


# ============================================================================
# Card Formatting Utilities
# ============================================================================


def format_card(card_idx: int) -> Text:
    """Convert card index to rich Text with colored suit (e.g., 'A♠', 'K♥')."""
    if card_idx < 0:
        return Text("??", style="dim")

    rank = card_idx % c.NUM_RANKS
    suit = card_idx // c.NUM_RANKS

    rank_str = RANK_NAMES[rank]
    suit_str = SUIT_SYMBOLS[suit]
    suit_color = SUIT_COLORS[suit]

    text = Text()
    text.append(rank_str, style="bold")
    text.append(suit_str, style=f"bold {suit_color}")
    return text


def format_cards(card_indices: list[int]) -> Text:
    """Format multiple cards with spacing."""
    text = Text()
    for i, idx in enumerate(card_indices):
        if i > 0:
            text.append(" ")
        text.append(format_card(idx))
    return text


def parse_observation_cards(obs: dict, env_idx: int = 0) -> dict[str, list[int]]:
    """Parse observation cards into hole cards and community cards."""
    cards = obs["cards"][env_idx].cpu().tolist()

    hole = cards[0:2]
    community_raw = cards[2:7]
    community = [c for c in community_raw if c >= 0]  # Filter undealt cards

    return {"hole": hole, "community": community}


# ============================================================================
# Snapshot Loading
# ============================================================================


def load_snapshot_model(
    path: str, device: torch.device, scalar_dim: int, config: dict
) -> PokerPolicyNet:
    """
    Load a snapshot model from checkpoint.
    Based on train/train.py:106-125
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)

    # Clean state dict if from compiled model (removes '_orig_mod.' prefix)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}

    # Create network with config
    net = PokerPolicyNet(
        scalar_dim=scalar_dim,
        card_embed_dim=config.get("card_embed_dim", 64),
        mlp_dim=config.get("mlp_dim", 256),
        torso_layers=config.get("torso_layers", 2),
        lstm_hidden=config.get("lstm_hidden", 256),
        head_layers=config.get("head_layers", 2),
        head_dim=config.get("head_dim", 128),
    ).to(device=device)

    net.load_state_dict(state_dict)
    net.eval()
    return net


def load_latest_snapshot(pool_dir: Path, device: torch.device, scalar_dim: int) -> tuple:
    """Load the most recent snapshot from the league pool."""
    if not pool_dir.exists():
        console.print(
            f"[red]Error: Pool directory not found: {pool_dir}[/red]\n"
            "Please train an agent first or specify a valid --pool-dir",
            style="bold",
        )
        raise FileNotFoundError(f"Pool directory not found: {pool_dir}")

    pool = UnifiedOpponentPool(pool_dir=pool_dir, bot_names=[])
    pool.load_index()
    snapshots = pool.get_snapshots()

    if not snapshots:
        console.print(
            "[red]Error: No snapshots found in league pool[/red]\n"
            "Please train an agent first to create snapshots.",
            style="bold",
        )
        raise ValueError("No snapshots found in league pool")

    # Get latest snapshot (highest env_steps, then update)
    latest = max(snapshots, key=lambda m: (m.env_steps or 0, m.update or 0))

    # Default config (can be customized if needed)
    config = {
        "card_embed_dim": 64,
        "mlp_dim": 256,
        "torso_layers": 2,
        "lstm_hidden": 256,
        "head_layers": 2,
        "head_dim": 128,
    }

    snapshot_path = pool_dir / latest.id
    net = load_snapshot_model(str(snapshot_path), device, scalar_dim, config)

    return net, latest


# ============================================================================
# Game State Display
# ============================================================================


def get_stage_name(obs: dict, env_idx: int = 0) -> str:
    """Get current game stage name from observation."""
    # Extract stage from scalars[7] (stage / 5.0)
    stage_normalized = obs["scalars"][env_idx, 7].item()
    stage = int(stage_normalized * 5.0)
    return STAGE_NAMES.get(stage, "Unknown")


def get_pot_size(obs: dict, env_idx: int = 0) -> int:
    """Estimate pot size from observation."""
    # scalars[5] = pot / (starting_stack * 2)
    # We'll need to reverse this, but we don't have starting_stack in obs
    # So we'll compute from big_blind and effective stack depth
    big_blind = obs["big_blind"][env_idx].item()
    # Assume starting stack is 100 BB by default
    starting_stack = 100 * big_blind

    pot_normalized = obs["scalars"][env_idx, 5].item()
    pot = int(pot_normalized * (starting_stack * 2))
    return pot


def get_player_stacks_and_bets(obs: dict, env_idx: int = 0) -> tuple:
    """Extract stack sizes and current bets from observation."""
    big_blind = obs["big_blind"][env_idx].item()
    starting_stack = 100 * big_blind  # Assumption

    # scalars[1] = player_stack / starting_stack
    # scalars[2] = opponent_stack / starting_stack
    player_stack = int(obs["scalars"][env_idx, 1].item() * starting_stack)
    opp_stack = int(obs["scalars"][env_idx, 2].item() * starting_stack)

    # scalars[3] = player_bet / pot_total (harder to reverse)
    # scalars[4] = opponent_bet / pot_total
    # We'll estimate from to_call and pot
    pot = get_pot_size(obs, env_idx)

    # scalars[6] = to_call / player_stack
    to_call = int(obs["scalars"][env_idx, 6].item() * player_stack)

    # Current player has to_call amount less in the pot
    # This is approximate - exact tracking requires game state
    player_bet = max(0, pot // 2 - to_call)
    opp_bet = max(0, pot // 2)

    return player_stack, opp_stack, player_bet, opp_bet


def display_game_state(
    obs: dict,
    env_idx: int,
    hand_num: int,
    human_seat: int,
    agent_name: str,
    last_action: str | None = None,
    show_hole_cards: bool = True,
) -> None:
    """Display current game state using rich tables and panels."""
    console.clear()

    # Title
    stage_name = get_stage_name(obs, env_idx)
    title = Panel(
        f"[bold cyan]Hand #{hand_num} - {stage_name}[/bold cyan]",
        border_style="cyan",
    )
    console.print(title)
    console.print()

    # Get game info
    big_blind = obs["big_blind"][env_idx].item()
    player_stack, opp_stack, player_bet, opp_bet = get_player_stacks_and_bets(obs, env_idx)
    pot = get_pot_size(obs, env_idx)

    # Position labels
    human_position = "Button/SB" if human_seat == 0 else "Big Blind"
    agent_position = "Big Blind" if human_seat == 0 else "Button/SB"

    # Player info table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Player", style="bold")
    table.add_column("Position")
    table.add_column("Stack", justify="right")
    table.add_column("Bet", justify="right")

    # Display in order: human first
    table.add_row(
        "[green]You[/green]",
        human_position,
        f"[green]{player_stack}[/green] chips",
        f"[yellow]{player_bet}[/yellow] chips" if player_bet > 0 else "-",
    )
    table.add_row(
        f"[red]{agent_name}[/red]",
        agent_position,
        f"[red]{opp_stack}[/red] chips",
        f"[yellow]{opp_bet}[/yellow] chips" if opp_bet > 0 else "-",
    )

    console.print(table)
    console.print()

    # Pot
    pot_display = Text()
    pot_display.append("POT: ", style="bold")
    pot_display.append(f"{pot} chips", style="bold blue")
    pot_display.append(f" ({pot // big_blind:.1f} BB)", style="dim")
    console.print(pot_display, justify="center")
    console.print()

    # Cards
    cards_data = parse_observation_cards(obs, env_idx)

    # Community cards
    if cards_data["community"]:
        community_text = Text("Board: ", style="bold yellow")
        community_text.append(format_cards(cards_data["community"]))
        console.print(community_text, justify="center")
        console.print()

    # Hole cards (only show if it's human's turn or show_hole_cards is True)
    if show_hole_cards:
        hole_text = Text("Your cards: ", style="bold green")
        hole_text.append(format_cards(cards_data["hole"]))
        console.print(hole_text)
        console.print()

    # Last action
    if last_action:
        console.print(f"[dim]Last action: {last_action}[/dim]")
        console.print()


def display_showdown(
    obs: dict,
    env_idx: int,
    human_seat: int,
    agent_seat: int,
    human_cards: list[int],
    agent_cards: list[int],
    winner: int,
    chips_won: int,
) -> None:
    """Display showdown with both hands revealed."""
    cards_data = parse_observation_cards(obs, env_idx)

    console.print()
    console.print("[bold yellow]═══ SHOWDOWN ═══[/bold yellow]", justify="center")
    console.print()

    # Show both hands
    human_text = Text("Your hand: ", style="bold green")
    human_text.append(format_cards(human_cards))
    console.print(human_text)

    agent_text = Text("Agent hand: ", style="bold red")
    agent_text.append(format_cards(agent_cards))
    console.print(agent_text)

    console.print()
    console.print(Text("Board: ", style="bold yellow") + format_cards(cards_data["community"]))
    console.print()

    # Winner
    if winner == human_seat:
        console.print(f"[bold green]You win {chips_won} chips![/bold green]")
    elif winner == agent_seat:
        console.print(f"[bold red]Agent wins {chips_won} chips![/bold red]")
    else:
        console.print(f"[bold yellow]Split pot: {chips_won} chips each[/bold yellow]")

    console.print()


# ============================================================================
# Human Input Handler
# ============================================================================


def get_legal_actions(obs: dict, env_idx: int = 0) -> list[int]:
    """Get list of legal action indices."""
    action_mask = obs["action_mask"][env_idx]
    return [i for i in range(c.NUM_ACTIONS) if action_mask[i]]


def parse_human_input(user_input: str, obs: dict, env_idx: int = 0) -> tuple[int, int] | None:
    """
    Parse human input string into (action_type, raise_amount).
    Returns None if input is invalid.
    """
    parts = user_input.lower().strip().split()
    if not parts:
        return None

    cmd = parts[0]

    # Map command to action
    action_map = {
        "f": c.ACTION_FOLD,
        "fold": c.ACTION_FOLD,
        "k": c.ACTION_CHECK,
        "check": c.ACTION_CHECK,
        "c": c.ACTION_CALL,
        "call": c.ACTION_CALL,
        "r": c.ACTION_RAISE,
        "raise": c.ACTION_RAISE,
    }

    if cmd not in action_map:
        return None

    action_type = action_map[cmd]

    # Validate against action mask
    if not obs["action_mask"][env_idx, action_type]:
        console.print(f"[red]Illegal action: {ACTION_NAMES[action_type]} is not allowed[/red]")
        return None

    # Handle raise amount
    raise_amount = 0
    if action_type == c.ACTION_RAISE:
        min_raise = obs["min_raise"][env_idx].item()
        max_raise = obs["max_raise"][env_idx].item()

        # Parse raise amount if provided
        if len(parts) > 1:
            try:
                raise_amount = int(parts[1])
            except ValueError:
                console.print(f"[red]Invalid raise amount: {parts[1]}[/red]")
                return None
        else:
            # Ask for amount
            console.print(f"Raise range: {min_raise} to {max_raise} chips (delta, not total bet)")
            amount_str = Prompt.ask("Raise amount (chips)")
            try:
                raise_amount = int(amount_str)
            except ValueError:
                console.print(f"[red]Invalid raise amount: {amount_str}[/red]")
                return None

        # Validate raise bounds
        if (raise_amount < min_raise or raise_amount > max_raise) and raise_amount != max_raise:
            console.print(
                f"[red]Raise amount {raise_amount} is out of range [{min_raise}, {max_raise}][/red]"
            )
            return None

    return action_type, raise_amount


def get_human_action(obs: dict, env_idx: int = 0) -> tuple[int, int]:
    """Get and validate human action input."""
    legal = get_legal_actions(obs, env_idx)

    # Show legal actions
    action_text = Text("Legal actions: ", style="bold")
    action_strs = []
    for action_idx in legal:
        name = ACTION_NAMES[action_idx].lower()
        action_strs.append(f"[{name[0]}]{name[1:]}")

    action_text.append(" ".join(action_strs), style="cyan")
    console.print(action_text)
    console.print()

    # Show raise bounds if raise is legal
    if c.ACTION_RAISE in legal:
        min_raise = obs["min_raise"][env_idx].item()
        max_raise = obs["max_raise"][env_idx].item()
        console.print(f"[dim]Raise range: {min_raise} - {max_raise} chips (delta)[/dim]")
        console.print()

    # Get input
    while True:
        user_input = Prompt.ask("[bold green]Your action[/bold green]")
        result = parse_human_input(user_input, obs, env_idx)
        if result is not None:
            return result
        console.print("[yellow]Invalid input. Try: f/fold, k/check, c/call, r/raise[/yellow]\n")


# ============================================================================
# Agent Action Display
# ============================================================================


def format_agent_action(action_type: int, raise_amount: int, to_call: int) -> str:
    """Format agent action as a readable string."""
    action_name = ACTION_NAMES[action_type]

    if action_type == c.ACTION_RAISE:
        total_bet = to_call + raise_amount
        return f"{action_name} {raise_amount} chips (total bet: {total_bet})"
    elif action_type == c.ACTION_CALL:
        return f"{action_name} {to_call} chips"
    else:
        return action_name


# ============================================================================
# Game Session
# ============================================================================


class GameSession:
    """Manages a poker session with hand tracking and stats."""

    def __init__(
        self,
        env: WarpPokerEnv,
        agent_net: PokerPolicyNet,
        human_seat: int,
        agent_meta,
        device: torch.device,
    ):
        self.env = env
        self.agent_net = agent_net
        self.human_seat = human_seat
        self.agent_seat = 1 - human_seat
        self.agent_meta = agent_meta
        self.device = device

        # Agent name for display
        env_steps = agent_meta.env_steps or 0
        update = agent_meta.update or 0
        self.agent_name = f"Agent [{env_steps // 1000000}M env, {update // 1000}K upd]"

        # Session stats
        self.hands_played = 0
        self.human_chips_won = 0

        # LSTM state for agent
        self.h, self.c = PokerPolicyNet.init_state(1, self.agent_net.lstm_hidden, self.device)

        # For showdown tracking
        self.starting_stacks = None

    def play_hand(self) -> dict:
        """Play one complete hand and return results."""
        self.hands_played += 1

        # Reset environment
        obs = self.env.reset()
        env_idx = 0

        # Track initial stacks for chips won calculation
        big_blind = obs["big_blind"][env_idx].item()
        starting_stack = 100 * big_blind
        initial_human_stack = starting_stack

        # Store cards for showdown
        human_hole_cards = None
        agent_hole_cards = None

        last_action_str = None
        terminated = False

        while not terminated:
            player_id = obs["player_id"][env_idx].item()

            # Determine whose turn
            if player_id == self.human_seat:
                # Human's turn
                # Store human's hole cards when we first see them
                if human_hole_cards is None:
                    cards_data = parse_observation_cards(obs, env_idx)
                    human_hole_cards = cards_data["hole"]

                display_game_state(
                    obs,
                    env_idx,
                    self.hands_played,
                    self.human_seat,
                    self.agent_name,
                    last_action_str,
                    show_hole_cards=True,
                )

                action_type, raise_amount = get_human_action(obs, env_idx)

                # Convert to tensor
                action_tensor = torch.tensor([action_type], dtype=torch.int32, device=self.device)
                amount_tensor = torch.tensor(
                    [[raise_amount]], dtype=torch.int32, device=self.device
                )

                # Format action for display
                to_call = int(
                    obs["scalars"][env_idx, 6].item()
                    * obs["scalars"][env_idx, 1].item()
                    * starting_stack
                )
                last_action_str = f"You: {format_agent_action(action_type, raise_amount, to_call)}"

            else:
                # Agent's turn
                # Store agent's hole cards (we won't show them until showdown)
                if agent_hole_cards is None:
                    cards_data = parse_observation_cards(obs, env_idx)
                    agent_hole_cards = cards_data["hole"]

                # Run agent policy
                with torch.no_grad():
                    policy_out = self.agent_net.forward_step(
                        cards=obs["cards"],
                        scalars=obs["scalars"],
                        action_mask=obs["action_mask"],
                        h=self.h,
                        c=self.c,
                        terminated=obs["terminated"],
                        deterministic=False,
                    )

                # Update LSTM state
                self.h, self.c = policy_out.h, policy_out.c

                action_type = policy_out.action_type[0].item()
                raise_frac = policy_out.raise_frac[0]

                # Convert raise_frac to amount
                amounts = raise_frac_to_amount(
                    raise_frac=raise_frac,
                    min_raise=obs["min_raise"],
                    max_raise=obs["max_raise"],
                )
                amounts = torch.where(
                    policy_out.action_type.eq(c.ACTION_RAISE),
                    amounts,
                    torch.zeros_like(amounts),
                )

                action_tensor = policy_out.action_type.to(dtype=torch.int32)
                amount_tensor = amounts.to(dtype=torch.int32)

                # Format action for display
                to_call = int(
                    obs["scalars"][env_idx, 6].item()
                    * obs["scalars"][env_idx, 1].item()
                    * starting_stack
                )
                raise_amt = amounts[0].item()
                last_action_str = (
                    f"{self.agent_name}: {format_agent_action(action_type, raise_amt, to_call)}"
                )

                # Display agent's action (but not hole cards)
                display_game_state(
                    obs,
                    env_idx,
                    self.hands_played,
                    self.human_seat,
                    self.agent_name,
                    last_action_str,
                    show_hole_cards=False,
                )

                console.print(f"[bold red]{last_action_str}[/bold red]")
                time.sleep(1.0)  # Pause for readability

            # Step environment
            obs = self.env.step(action_tensor, amount_tensor)
            terminated = obs["terminated"][env_idx].item()

        # Hand is over - check rewards
        final_stack = int(obs["scalars"][env_idx, 1].item() * starting_stack)
        chips_won = final_stack - initial_human_stack
        self.human_chips_won += chips_won

        # Determine winner
        if chips_won > 0:
            winner = self.human_seat
        elif chips_won < 0:
            winner = self.agent_seat
        else:
            winner = -1  # Split pot

        # Show final state with showdown if we got there
        stage_name = get_stage_name(obs, env_idx)
        if stage_name in ["Showdown", "Terminal"] and human_hole_cards and agent_hole_cards:
            # Rotate agent cards to agent's perspective
            # Note: obs always shows from active player perspective, which might not be consistent
            # We stored the cards when each player first acted, which should be correct
            display_showdown(
                obs,
                env_idx,
                self.human_seat,
                self.agent_seat,
                human_hole_cards,
                agent_hole_cards,
                winner,
                abs(chips_won),
            )
        else:
            # Just show final state (someone folded)
            display_game_state(
                obs,
                env_idx,
                self.hands_played,
                self.human_seat,
                self.agent_name,
                last_action_str,
                show_hole_cards=True,
            )
            console.print()
            if chips_won > 0:
                console.print(f"[bold green]You win {chips_won} chips![/bold green]")
            elif chips_won < 0:
                console.print(f"[bold red]Agent wins {abs(chips_won)} chips![/bold red]")
            console.print()

        # Reset LSTM state for new hand
        self.h, self.c = PokerPolicyNet.init_state(1, self.agent_net.lstm_hidden, self.device)

        return {
            "winner": winner,
            "chips_won": chips_won,
            "human_hole": human_hole_cards,
            "agent_hole": agent_hole_cards,
        }

    def run_session(self) -> None:
        """Main loop: play hands until user quits."""
        # Welcome screen
        console.clear()
        seat_label = "Button/SB (P0)" if self.human_seat == 0 else "Big Blind (P1)"
        welcome = Panel(
            "[bold cyan]Welcome to PokerGPU Interactive Play![/bold cyan]\n\n"
            f"You are playing against: [bold]{self.agent_name}[/bold]\n"
            f"Your seat: [green]{seat_label}[/green]\n\n"
            "[dim]Commands: f/fold, k/check, c/call, r/raise <amount>[/dim]",
            border_style="cyan",
        )
        console.print(welcome)
        console.print()
        input("Press Enter to start...")

        while True:
            # Play one hand
            self.play_hand()

            # Show session stats
            console.print()
            console.print("[bold]Session stats:[/bold]")
            console.print(f"  Hands played: {self.hands_played}")
            console.print(f"  Net chips won: {self.human_chips_won:+d}")
            if self.hands_played > 0:
                bb_per_hand = self.human_chips_won / self.hands_played / 2  # BB = 2
                console.print(f"  BB/hand: {bb_per_hand:+.2f}")
            console.print()

            # Ask to continue
            response = Prompt.ask(
                "Continue? ([y]es / [n]o / [s]wap seats)",
                choices=["y", "n", "s", "yes", "no", "swap"],
                default="y",
            )

            if response in ["n", "no"]:
                break
            elif response in ["s", "swap"]:
                # Swap seats
                self.human_seat = 1 - self.human_seat
                self.agent_seat = 1 - self.agent_seat
                console.print(
                    f"[yellow]Swapped seats! You are now "
                    f"{'Button/SB (P0)' if self.human_seat == 0 else 'Big Blind (P1)'}[/yellow]"
                )
                console.print()
                time.sleep(1.5)

        # Session summary
        console.clear()
        summary = Panel(
            "[bold cyan]Session Complete![/bold cyan]\n\n"
            f"Hands played: {self.hands_played}\n"
            f"Net chips won: {self.human_chips_won:+d}\n"
            + (
                f"BB/hand: {self.human_chips_won / self.hands_played / 2:+.2f}\n"
                if self.hands_played > 0
                else ""
            )
            + "\nThanks for playing!",
            border_style="cyan",
        )
        console.print(summary)


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Play interactive heads-up poker against a trained agent"
    )
    parser.add_argument(
        "--pool-dir",
        type=str,
        default="checkpoints/league_pool",
        help="League pool directory containing snapshots",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument(
        "--seat",
        type=str,
        choices=["0", "1", "random"],
        default="random",
        help="Your seat position (0=Button/SB, 1=BB, random=random choice)",
    )
    parser.add_argument(
        "--starting-stack",
        type=int,
        default=200,
        help="Starting stack size in chips (default: 200)",
    )
    parser.add_argument(
        "--big-blind",
        type=int,
        default=2,
        help="Big blind size (default: 2)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Initialize Warp
    wp.init()

    # Create environment (num_envs=1 for interactive play)
    env = WarpPokerEnv(
        num_envs=1,
        device=args.device,
        starting_stack=args.starting_stack,
        big_blind=args.big_blind,
    )

    # Get scalar dim from environment
    obs = env.reset()
    scalar_dim = obs["scalars"].shape[1]

    # Load latest snapshot
    console.print("[cyan]Loading latest agent from league pool...[/cyan]")
    agent_net, agent_meta = load_latest_snapshot(Path(args.pool_dir), device, scalar_dim)
    console.print("[green]Agent loaded successfully![/green]\n")

    # Determine seat
    human_seat = random.choice([0, 1]) if args.seat == "random" else int(args.seat)

    # Start session
    session = GameSession(env, agent_net, human_seat, agent_meta, device)
    session.run_session()


if __name__ == "__main__":
    main()
