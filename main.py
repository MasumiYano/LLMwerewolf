import os
from game_rag import GameRAG
from Player import PlayerStatus, GameState  # Import GameState from Player
from controller import Controller
from config import PLAYER_NUM, VILLAGER_NUM, WEREWOLF_NUM


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY")
        return

    rag = GameRAG()

    werewolf_strategies = """
    Werewolf strategies:
        1. During night discussion, coordinate with teammates on targets.
        2. During day discussion, blend in and deflect suspicion
        3. Target influential villagers or those who suspect you
        4. Create reasonable doubt about other players
        5. Build alliances with villagers 
        6. Vote strategically to avoid suspicion
    """
    villager_strategies = """
    Villager strategies:
        1. Listen carefully to all statements for inconsistencies
        2. Track who votes for whom across multiple days
        3. Pay attention to who deflects or changes topics
        4. Look for players who benefit from eliminations
        5. Ask probing questions during discussions
        6. Form alliances with trusted players
        7. Share your deductions openly but thoughtfully
    """

    game_state = GameState({
        "phase": "setup",
        "day_count": 0,
        "players": {},
        "alive_players": [],
        "last_eliminated": "",
        "last_night_victim": "",
    })

    rag.add_werewolf_knowledge(werewolf_strategies, game_state)
    rag.add_villager_knowledge(villager_strategies, game_state)

    game = Controller(rag, game_state)
    players = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Carlos"]
    game.setup_game(players)
    game.play_game()


if __name__ == "__main__":
    main()
