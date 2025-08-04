from game_rag import GameRAG
from Player import Player, GameState, PlayerStatus
from typing import Dict, List
import random
from werewolf import Werewolf
from villager import Villager
from langchain_core.messages import SystemMessage, HumanMessage
from config import VILLAGER_NUM, PLAYER_NUM, WEREWOLF_NUM, MAX_DISCUSSION_CYCLE


class Controller:
    def __init__(self, rag: GameRAG, game_state: GameState):
        self.rag = rag
        self.players: Dict[str, Player] = {}
        self.player_order: List[str] = []
        self.game_state: GameState = game_state

    def add_player(self, player: Player):
        """Add player to the game"""
        self.players[player.get_user_id()] = player
        self.game_state["players"][player.get_user_id()] = (
            PlayerStatus.ALIVE
        )  # Fixed missing ()
        self.game_state["alive_players"].append(player.get_user_id())

    def setup_game(self, player_names: List[str]):
        """Setup game with given player number"""
        if len(player_names) != PLAYER_NUM:  # Fixed number
            raise ValueError(f"Need exactly {PLAYER_NUM} players")

        werewolf_players = random.sample(player_names, WEREWOLF_NUM)
        self.player_order = player_names.copy()
        random.shuffle(self.player_order)

        print("==== GAME SETUP ====")
        for name in player_names:
            if name in werewolf_players:
                self.add_player(Werewolf(name, self.rag))
                print(f"{name} is werewolf")
            else:
                self.add_player(Villager(name, self.rag))
                print(f"{name} is a villager")

        print(f"\nDiscussion order: {' -> '.join(self.player_order)}")

    def werewolf_night_discussion(self):
        print("\n--- Werewolf Discussion ---")

        alive_werewolves = [
            player_id
            for player_id in self.game_state["alive_players"]
            if self.players[player_id].role_name == "werewolf"  # Fixed method call
        ]

        if len(alive_werewolves) < 2:
            werewolf = self.players[alive_werewolves[0]]
            target = werewolf.get_night_action(self.game_state)
            if target:
                print(
                    f"Remaining werewolf {alive_werewolves[0]} chooses to eliminate {target}"  # Fixed space
                )

            return target

        werewolf_discussion = []

        for werewolf_id in alive_werewolves:
            werewolf = self.players[werewolf_id]
            teammates = [w for w in alive_werewolves if w != werewolf_id]

            response = werewolf.discuss_night_target(
                self.game_state, teammates, werewolf_discussion
            )

            werewolf_discussion.append({"player": werewolf_id, "message": response})

            print(f"{werewolf_id}: {response}")

        print("\n--- Final Decision ---")
        werewolf_votes = {}

        for werewolf_id in alive_werewolves:
            werewolf = self.players[werewolf_id]

            discussion_context = "\n".join([
                f"{stmt['player']}: {stmt['message']}" for stmt in werewolf_discussion
            ])

            potential_targets = [
                p
                for p in self.game_state["alive_players"]
                if self.players[p].role_name != "werewolf"  # Fixed method call
            ]

            system_prompt = f"""Based on your team discussion, make your final choice for who to eliminate.
            
            Discussion summary:
                {discussion_context}

            Choose one player from: {potential_targets}

            Respond with just the player name.
            """

            config = {
                "configurable": {
                    "thread_id": f"werewolf_vote_{werewolf_id}_night_{self.game_state['day_count']}"
                }
            }

            messages = [  # Fixed variable name
                SystemMessage(content=system_prompt),
                HumanMessage(content="Your final vote?"),
            ]

            for event in werewolf.agent_executor.stream(
                {"messages": messages}, config=config, stream_mode="values"
            ):
                vote_response = event["messages"][-1].content

            target = werewolf._extract_target(
                vote_response, self.game_state["alive_players"]
            )

            if (
                target and self.players[target].role_name != "werewolf"
            ):  # Fixed method call
                werewolf_votes[werewolf_id] = target
                print(f"{werewolf_id} votes to eliminate {target}")

        if werewolf_votes:  # Fixed indentation
            votes = list(werewolf_votes.values())
            victim = max(set(votes), key=votes.count)
            return victim

        return None

    def night_phase(self):
        """Execute night phase with werewolf discussion"""
        print(f"\n{'=' * 50}")
        print(f"NIGHT {self.game_state['day_count']}")
        print(f"{'=' * 50}")
        self.game_state["phase"] = "night"

        victim = self.werewolf_night_discussion()

        if victim:
            self.eliminate_player(victim)
            self.game_state["last_night_victim"] = victim
            print(f"\n{victim} was eliminated during the night")
        else:
            print("\nNo one was eliminated tonight.")
            self.game_state["last_night_victim"] = ""

    def vote_to_continue_discussion(self, cycle_num: int) -> bool:
        """Ask all players if they want to continue discussion or move to voting"""
        print(f"\n--- DISCUSSION CONTINUATION VOTE (Cycle {cycle_num}) ---")

        votes = {}
        alive_in_order = [
            p for p in self.player_order if p in self.game_state["alive_players"]
        ]

        for player_id in alive_in_order:
            player = self.players[player_id]

            # Simple prompt asking if they want to continue discussion
            system_prompt = f"""DISCUSSION CONTINUATION VOTE - Day {self.game_state["day_count"]}, After Cycle {cycle_num}

            Game State:
                - Alive players: {self.game_state["alive_players"]}
                - Discussion cycles completed: {cycle_num}
                - Maximum cycles allowed: {MAX_DISCUSSION_CYCLE}

            Task: Decide if you want to continue discussion or move to voting phase.
            Respond with either "continue discussion" or "move to voting".
            """

            config = {
                "configurable": {
                    "thread_id": f"{player.role_name}_{player_id}_continue_vote_day_{self.game_state['day_count']}_cycle_{cycle_num}"
                }
            }

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content="Do you want to continue discussion or move to voting?"
                ),
            ]

            for event in player.agent_executor.stream(
                {"messages": messages}, config=config, stream_mode="values"
            ):
                response = event["messages"][-1].content

            # Extract the decision
            response_lower = response.lower()
            if "move to voting" in response_lower or "voting" in response_lower:
                vote = "voting"
            elif "continue" in response_lower:
                vote = "continue"
            else:
                # Default to continue if unclear
                vote = "continue"

            votes[player_id] = vote
            print(f"{player_id}: {vote}")

        # Count votes
        continue_votes = sum(1 for vote in votes.values() if vote == "continue")
        voting_votes = sum(1 for vote in votes.values() if vote == "voting")

        print(
            f"\nVote results: Continue Discussion: {continue_votes}, Move to Voting: {voting_votes}"
        )

        # Majority wants to move to voting
        if voting_votes > continue_votes:
            print("Majority voted to move to voting phase.")
            return False
        else:
            print("Majority voted to continue discussion.")
            return True

    def day_discussion(self):
        """Dynamic discussion with voting to continue after each cycle"""
        print(f"\n{'=' * 50}")
        print(f"DAY {self.game_state['day_count']} - Discussion")
        print(f"{'=' * 50}")
        self.game_state["phase"] = "day"

        all_statements = []
        alive_in_order = [
            p for p in self.player_order if p in self.game_state["alive_players"]
        ]

        cycle_num = 1

        while cycle_num <= MAX_DISCUSSION_CYCLE:
            print(f"\n--- DISCUSSION CYCLE: {cycle_num} ---")

            # One clockwise round - everyone speaks once
            for player_id in alive_in_order:
                player = self.players[player_id]

                if player.role_name == "werewolf":
                    teammates = self.get_werewolf_teammate(player_id)
                    statement = player.speak_in_discussion(
                        self.game_state, cycle_num, all_statements, teammates
                    )
                else:
                    statement = player.speak_in_discussion(
                        self.game_state, cycle_num, all_statements
                    )

                all_statements.append({
                    "player": player_id,
                    "message": statement,
                    "cycle": cycle_num,
                })

                print(f"{player_id}: {statement}")

            # After each cycle, vote on whether to continue
            if cycle_num >= MAX_DISCUSSION_CYCLE:
                print(
                    f"\nMaximum discussion cycles ({MAX_DISCUSSION_CYCLE}) reached. Moving to voting."
                )
                break

            # Ask players if they want to continue discussion
            should_continue = self.vote_to_continue_discussion(cycle_num)

            if not should_continue:
                break

            cycle_num += 1

        # Store conversation history
        discussion_summary = {
            stmt["player"]: stmt["message"] for stmt in all_statements
        }
        self.rag.add_conversations(discussion_summary, self.game_state)

        return all_statements

    def voting_phase(self, discussion_history: List[Dict[str, str]]):
        """Execute voting phase after discussion"""
        print("\n--- VOTING PHASE ---")
        self.game_state["phase"] = "voting"

        votes = {}
        alive_in_order = [
            p for p in self.player_order if p in self.game_state["alive_players"]
        ]

        for player_id in alive_in_order:
            player = self.players[player_id]

            if player.role_name == "villager":
                vote = player.get_vote(self.game_state, discussion_history)
            else:
                teammates = self.get_werewolf_teammate(player_id)
                vote = player.get_vote(self.game_state, teammates)

            if vote and vote in self.game_state["alive_players"] and vote != player_id:
                votes[player_id] = vote
                print(f"{player_id} votes for {vote}")
            else:
                print(f"{player_id} abstains")

        if votes:
            vote_counts = {}
            for vote in votes.values():
                vote_counts[vote] = vote_counts.get(vote, 0) + 1

            max_votes = max(vote_counts.values())
            candidates = [
                player for player, count in vote_counts.items() if count == max_votes
            ]
            eliminated = random.choice(candidates)

            print(f"\nVote results: {vote_counts}")
            self.eliminate_player(eliminated)
            self.game_state["last_eliminated"] = eliminated
            print(f"{eliminated} was voted out")
            print(f"{eliminated} was a {self.players[eliminated].role_name}")

        else:
            print("No votes cast - no elimination today.")
            self.game_state["last_eliminated"] = ""

    def eliminate_player(self, player_id: str):
        """Remove player from game"""
        self.game_state["players"][player_id] = PlayerStatus.DEAD
        self.game_state["alive_players"].remove(player_id)

    def check_game_end(self):
        """Check if game has ended and return winner"""
        alive_players = self.game_state["alive_players"]
        werewolves = [
            p for p in alive_players if self.players[p].role_name == "werewolf"
        ]
        villagers = [
            p for p in alive_players if self.players[p].role_name == "villager"
        ]

        if not werewolves:
            return "villagers"
        elif len(werewolves) >= len(villagers):
            return "werewolves"
        return None

    def play_game(self):
        print("\nStarting Werewolf Game...")
        print(f"{VILLAGER_NUM} Villagers vs {WEREWOLF_NUM} Werewolves")  # Fixed comment
        print(f"Maximum discussion cycles per day: {MAX_DISCUSSION_CYCLE}")

        while True:
            self.night_phase()

            winner = self.check_game_end()
            if winner:
                print(f"{winner.upper()} WIN")
                break

            discussion_history = self.day_discussion()
            self.voting_phase(discussion_history)

            winner = self.check_game_end()
            if winner:
                print(f"{winner.upper()} WIN")
                break

            self.game_state["day_count"] += 1
            print(f"Survivors: {self.game_state['alive_players']}")

        print(f"\nFinal survivors: {self.game_state['alive_players']}")
        for player_id, player in self.players.items():
            status = (
                "ALIVE" if player_id in self.game_state["alive_players"] else "DEAD"
            )
            print(f"{player_id}: {player.role_name} - {status}")

        self.rag.clear_conversation_history()

    def get_werewolf_teammate(self, player_id: str):
        """Get list of werewolf teammates for a given player"""
        if self.players[player_id].role_name != "werewolf":
            return []

        werewolf_teammates = [
            p_id
            for p_id in self.game_state["alive_players"]
            if self.players[p_id].role_name == "werewolf" and p_id != player_id
        ]

        return werewolf_teammates
