#include "chess-simulator.h"
// disservin's lib. drop a star on his hard work!
// https://github.com/Disservin/chess-library
#include <chrono>

#include "chess.hpp"
#include <random>
using namespace ChessSimulator;

std::string ChessSimulator::Move(std::string fen, int timeLimitMs) {
  // create your board based on the board string following the FEN notation
  // search for the best move using minimax / monte carlo tree search /
  // alpha-beta pruning / ... try to use nice heuristics to speed up the search
  // and have better results return the best move in UCI notation you will gain
  // extra points if you create your own board/move representation instead of
  // using the one provided by the library
  startTime = std::chrono::high_resolution_clock::now();

  chess::Board board(fen);
  chess::Movelist moves;
  chess::movegen::legalmoves(moves, board);
  if(moves.size() == 0)
    return "";

  // EDIT THIS FOR DIFFERENT APPROACH— false for MCTS, true for minimax
  bool isMiniMax = true;


  if (isMiniMax) {
    bool isMax;
    if (board.sideToMove() == chess::Color::WHITE) { isMax = true; }
    else { isMax = false; }

    // random move to start comparison
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, moves.size() - 1);

    chess::Move move = moves[dist(gen)];
    Minimax(board, 15, 0, 10, isMax, timeLimitMs);

    return chess::uci::moveToUci(move);
  }
  else {
    return chess::uci::moveToUci(MonteCarlo(board, 10));
  }
}

int ChessSimulator::Minimax(chess::Board &board, int depth, int alpha, int beta, bool isMax, int timeLimitMs) {
  auto currentTime = std::chrono::high_resolution_clock::now();
  if (depth == 0 || (currentTime - startTime).count() >= timeLimitMs - (timeLimitMs / 10)) {
    return Score(board);
  }

  chess::Movelist moves;
  chess::movegen::legalmoves(moves, board);

  int index = 0;
  int reduction = std::ceil(moves.size() / 0.5);

  int bestValue = -1;

  for (chess::Move& move : moves) {
    if (index >= moves.size()) { return Score(board); }
    auto [reason, result] = board.isGameOver();
    if (result != chess::GameResult::NONE) { return Score(board); }

    // apply move, make new board, unmake move
    board.makeMove(move);
    chess::Board nextBoard(board);
    board.unmakeMove(move);

    // late move reduction
    if (index > reduction) {
      depth /= 2;
      if (depth <= 1) { depth = 0; }
    }
    index++;

    int value = Score(nextBoard, move);

    // max score
    if (isMax) {
      if (value > bestValue) {
        bestValue = value;
        bestMove = move;
      }
      alpha = std::max(alpha, bestValue);
    }
    // min score
    else {
      if (value < bestValue) {
        bestValue = value;
        bestMove = move;
      }
      beta = std::min(beta, bestValue);
    }

    if (beta <= alpha) { break; }

    // recursion
    if (depth > 1) {
      // pass in current alpha & beta for aspiration pruning
      Minimax(nextBoard, depth - 1, alpha, beta, !isMax, timeLimitMs);
    }
  }
  return bestValue;
}


chess::Move ChessSimulator::MonteCarlo(chess::Board& board, int depth) {
  MCTSNode* root = new MCTSNode(board);

  for (int i = 0; i < depth; i++) {
    MCTSNode* leaf = select(root);

    if (!leaf->state.inCheck()) {
      expand(leaf);
      leaf = leaf->children[0];
    }

    double result = simulate(leaf);

    backpropagate(leaf, result);
  }

  return bestChild(root)->move;
}

MCTSNode* ChessSimulator::select(MCTSNode* node) {
  while (!node->isLeaf()) {
    node = *std::max_element(node->children.begin(), node->children.end(), [](MCTSNode* a, MCTSNode* b)
      { return a->ucb() < b->ucb(); }
      );
  }
  return node;
}

void ChessSimulator::expand(MCTSNode* node) {
  chess::Movelist nextStates;
  chess::movegen::legalmoves(nextStates, node->state);

  for (const chess::Move& move : nextStates) {
    node->state.makeMove(move);
    chess::Board nextBoard(node->state);
    node->state.unmakeMove(move);
    MCTSNode* child = new MCTSNode(nextBoard);
    child->parent = node;
    node->children.push_back(child);
  }
}

double ChessSimulator::simulate(MCTSNode* node) {
  auto color = node->state.sideToMove();
  chess::Board state = node->state;
  chess::Move move = chess::Move();

  // non terminal state
  for (;;) {
    chess::Move lastMove = move;
    auto currentColor = state.sideToMove();

    auto [reason, result] = state.isGameOver();
    if (result == chess::GameResult::WIN) {
      if (currentColor == color) {
        node->move = move;
      }
      else {
        node->move = lastMove;
      }
      return 1.0;
    }
    if (result == chess::GameResult::LOSE) {
      if (currentColor == color) {
        node->move = move;
      }
      else {
        node->move = lastMove;
      }
      return 0.0;
    }
    if (result == chess::GameResult::DRAW) {
      if (currentColor == color) {
        node->move = move;
      }
      else {
        node->move = lastMove;
      }
      return 0.5;
    }
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, state);
    move = moves[rand() % moves.size()];
    state.makeMove(move);
  }
}

void ChessSimulator::backpropagate(MCTSNode* node, double result) {
  while (node != nullptr) {
    node->visits++;
    node->wins += result;
    result = 1.0 - result;
    node = node->parent;
  }
}

MCTSNode* ChessSimulator::bestChild(MCTSNode* root) {
  return *std::max_element(
      root->children.begin(), root->children.end(),
      [](MCTSNode* a, MCTSNode* b) {
          return a->visits < b->visits;
      }
  );
}


int ChessSimulator::Score(const chess::Board &board, const chess::Move& move) {
  int whiteScore = 0, blackScore = 0;

  auto wK = board.pieces(chess::PieceType::KING, chess::Color::WHITE);
  whiteScore += (wK.count() * 200);
  auto wQ = board.pieces(chess::PieceType::QUEEN, chess::Color::WHITE);
  whiteScore += (wQ.count() * 20);
  auto wR = board.pieces(chess::PieceType::ROOK, chess::Color::WHITE);
  whiteScore += (wR.count() * 15);
  auto wB = board.pieces(chess::PieceType::BISHOP, chess::Color::WHITE);
  whiteScore += (wB.count() * 10);
  auto wN = board.pieces(chess::PieceType::KNIGHT, chess::Color::WHITE);
  whiteScore += (wN.count() * 10);
  auto wP = board.pieces(chess::PieceType::PAWN, chess::Color::WHITE);
  whiteScore += wP.count();

  auto bK = board.pieces(chess::PieceType::KING, chess::Color::WHITE);
  blackScore += (bK.count() * 200);
  auto bQ = board.pieces(chess::PieceType::QUEEN, chess::Color::WHITE);
  blackScore += (bQ.count() * 20);
  auto bR = board.pieces(chess::PieceType::ROOK, chess::Color::WHITE);
  blackScore += (bR.count() * 15);
  auto bB = board.pieces(chess::PieceType::BISHOP, chess::Color::WHITE);
  blackScore += (bB.count() * 10);
  auto bN = board.pieces(chess::PieceType::KNIGHT, chess::Color::WHITE);
  blackScore += (bN.count() * 10);
  auto bP = board.pieces(chess::PieceType::PAWN, chess::Color::WHITE);
  blackScore += bP.count();

  // white castling
  if (board.getCastleString().contains("K") || board.getCastleString().contains("Q")) { whiteScore += 10; }
  // black castling
  if (board.getCastleString().contains("k") || board.getCastleString().contains("q")) { blackScore += 10; }

  if (board.sideToMove() == chess::Color::WHITE) {
    if (board.inCheck()) { whiteScore -= 400; }
    if (board.isCapture(move)) { whiteScore += 20; }
  }
  if (board.sideToMove() == chess::Color::BLACK) {
    if (board.inCheck()) { blackScore -= 400; }
    if (board.isCapture(move)) { blackScore += 20; }
  }

  if (board.sideToMove() == chess::Color::WHITE) { return whiteScore - blackScore; }
  return blackScore - whiteScore;
}
