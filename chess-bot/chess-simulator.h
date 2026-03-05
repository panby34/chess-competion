#pragma once
#include <chess.hpp>
#include <chrono>
#include <string>
#include <cmath>

namespace chess {
    class Board;
}

namespace ChessSimulator {

    class MCTSNode {
    public:
        MCTSNode* parent;
        std::vector<MCTSNode*> children;
        double wins;
        int visits;
        chess::Board state;
        chess::Move move;

        double ucb(double C = std::sqrt(2.0)) {
            if (visits == 0) { return 99999999; }
            return (wins / visits) + C * std::sqrt(std::log(parent->visits) / visits);
        }

        bool isLeaf() {
            return children.empty();
        }

        MCTSNode(chess::Board board) {
            parent = nullptr;
            children = std::vector<MCTSNode*>();
            wins = 0;
            visits = 0;
            state = board;
        }
    };

    MCTSNode* select(MCTSNode* node);
    void expand(MCTSNode* node);
    double simulate(MCTSNode* node);
    void backpropagate(MCTSNode* node, double result);
    MCTSNode* bestChild(MCTSNode* root);

/**
 * @brief Move a piece on the board
 *
 * @param fen The board as FEN
 * @return std::string The move as UCI
 */
    std::string Move(std::string fen, int timeLimitMs = 1000000);
    int Minimax(chess::Board& board, int depth, int alpha, int beta, bool isMax, int timeLimitMs);
    chess::Move MonteCarlo(chess::Board& board, int depth);
    int Score(const chess::Board& board, const chess::Move& move = chess::Move());

    inline chess::Move bestMove;
    inline auto startTime = std::chrono::high_resolution_clock::now();
} // namespace ChessSimulator
