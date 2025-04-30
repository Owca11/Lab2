import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from typing import Tuple, List, Optional, Dict


def read_fasta(file_path: str) -> Tuple[str, str]:
    """
    Read sequences from a FASTA file.

    Args:
        file_path: Path to the FASTA file.

    Returns:
        Tuple of two sequences (seq1, seq2).

    Raises:
        ValueError: If the file does not contain exactly two sequences.
        FileNotFoundError: If the specified file cannot be found.
    """
    try:
        with open(file_path, "r") as file:
            records = list(SeqIO.parse(file, "fasta"))
            if len(records) != 2:
                raise ValueError("FASTA file must contain exactly two sequences.")
            return str(records[0].seq), str(records[1].seq)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")


def initialize_matrix(seq1: str, seq2: str, gap_penalty: int) -> np.ndarray:
    """
    Initialize the scoring matrix with gap penalties.

    The matrix is initialized with increasing gap penalties in the first row and column,
    representing the cost of introducing gaps at the beginning of the alignment.

    Args:
        seq1: First sequence.
        seq2: Second sequence.
        gap_penalty: Penalty for introducing a gap (should be negative for most cases).

    Returns:
        Initialized matrix with gap penalties in the first row and column.
    """
    rows, cols = len(seq1) + 1, len(seq2) + 1
    matrix = np.zeros((rows, cols))
    for i in range(rows):
        matrix[i, 0] = i * gap_penalty
    for j in range(cols):
        matrix[0, j] = j * gap_penalty
    return matrix


def fill_matrix(
        seq1: str,
        seq2: str,
        matrix: np.ndarray,
        match_score: int,
        mismatch_penalty: int,
        gap_penalty: int,
) -> np.ndarray:
    """
    Fill the scoring matrix using the Needleman-Wunsch algorithm.

    The matrix is filled by considering three possible moves for each cell:
    diagonal (match/mismatch), vertical (gap in seq2), or horizontal (gap in seq1).

    Args:
        seq1: First sequence.
        seq2: Second sequence.
        matrix: Initialized scoring matrix.
        match_score: Score for matching characters (typically positive).
        mismatch_penalty: Penalty for mismatches (typically negative).
        gap_penalty: Penalty for gaps (typically negative).

    Returns:
        Filled scoring matrix with optimal scores for all possible alignments.
    """
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            match = matrix[i - 1, j - 1] + (match_score if seq1[i - 1] == seq2[j - 1] else mismatch_penalty)
            delete = matrix[i - 1, j] + gap_penalty
            insert = matrix[i, j - 1] + gap_penalty
            matrix[i, j] = max(match, delete, insert)
    return matrix


def traceback(
        seq1: str,
        seq2: str,
        matrix: np.ndarray,
        match_score: int,
        mismatch_penalty: int,
        gap_penalty: int,
) -> Tuple[str, str, List[Tuple[int, int]]]:
    """
    Perform traceback to find the optimal alignment path.

    Starting from the bottom-right corner of the matrix, the algorithm traces back
    to the top-left corner, choosing at each step the move that led to the current score.

    Args:
        seq1: First sequence.
        seq2: Second sequence.
        matrix: Filled scoring matrix.
        match_score: Score for matching characters.
        mismatch_penalty: Penalty for mismatches.
        gap_penalty: Penalty for gaps.

    Returns:
        Tuple containing:
        - aligned_seq1: First sequence with gaps inserted
        - aligned_seq2: Second sequence with gaps inserted
        - path: List of (i,j) coordinates representing the optimal path through the matrix
    """
    aligned_seq1, aligned_seq2 = [], []
    i, j = len(seq1), len(seq2)
    path = [(i, j)]

    while i > 0 or j > 0:
        if i > 0 and j > 0 and matrix[i, j] == matrix[i - 1, j - 1] + (
                match_score if seq1[i - 1] == seq2[j - 1] else mismatch_penalty
        ):
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and matrix[i, j] == matrix[i - 1, j] + gap_penalty:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append("-")
            i -= 1
        else:
            aligned_seq1.append("-")
            aligned_seq2.append(seq2[j - 1])
            j -= 1
        path.append((i, j))

    aligned_seq1 = "".join(reversed(aligned_seq1))
    aligned_seq2 = "".join(reversed(aligned_seq2))
    path = list(reversed(path))

    return aligned_seq1, aligned_seq2, path


def calculate_statistics(
        aligned_seq1: str,
        aligned_seq2: str,
        matrix: np.ndarray,
        match_score: int,
        mismatch_penalty: int,
        gap_penalty: int,
) -> Dict[str, float]:
    """
    Calculate alignment statistics including matches, mismatches, gaps, and percentages.

    Args:
        aligned_seq1: First aligned sequence with gaps.
        aligned_seq2: Second aligned sequence with gaps.
        matrix: Filled scoring matrix (used to get the final alignment score).
        match_score: Score for matches (used for verification).
        mismatch_penalty: Penalty for mismatches (used for verification).
        gap_penalty: Penalty for gaps (used for verification).

    Returns:
        Dictionary containing alignment statistics:
        - matches: Number of matching positions
        - mismatches: Number of mismatching positions
        - gaps: Number of gaps in alignment
        - alignment_length: Total length of alignment
        - identity: Percentage of identical positions
        - gap_percentage: Percentage of gaps in alignment
        - score: Final alignment score from the matrix
    """
    matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b and a != "-")
    mismatches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a != b and a != "-" and b != "-")
    gaps = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == "-" or b == "-")
    alignment_length = len(aligned_seq1)
    identity = (matches / alignment_length) * 100 if alignment_length > 0 else 0
    gap_percentage = (gaps / alignment_length) * 100 if alignment_length > 0 else 0

    return {
        "matches": matches,
        "mismatches": mismatches,
        "gaps": gaps,
        "alignment_length": alignment_length,
        "identity": identity,
        "gap_percentage": gap_percentage,
        "score": matrix[-1, -1],
    }


def plot_matrix(
        matrix: np.ndarray,
        seq1: str,
        seq2: str,
        path: List[Tuple[int, int]],
        output_file: Optional[str] = None,
) -> None:
    """
    Plot the scoring matrix with the optimal path.

    Creates a heatmap of the scoring matrix with the optimal path highlighted.
    The plot includes sequence labels and a colorbar indicating score values.

    Args:
        matrix: Filled scoring matrix.
        seq1: First sequence (for labeling).
        seq2: Second sequence (for labeling).
        path: Optimal path through the matrix.
        output_file: Optional path to save the plot. If None, displays the plot.

    Returns:
        None
    """
    plt.figure(figsize=(10, 8))
    cax = plt.imshow(matrix, cmap='inferno')
    plt.colorbar(cax)

    # Annotate matrix values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, int(matrix[i, j]), va='center', ha='center', color='black')

    # Plot the optimal path
    path_x, path_y = zip(*path)
    plt.plot(path_y, path_x, marker='', color='black', linestyle='-', linewidth=1)

    # X-axis (seq2) on top
    ax = plt.gca()
    ax.xaxis.tick_top()  # Move x-axis to top
    ax.xaxis.set_label_position('top')  # Move x-axis label to top

    # Set ticks and labels
    ax.set_xticks(np.arange(1, len(seq2) + 1))
    ax.set_yticks(np.arange(1, len(seq1) + 1))
    ax.set_xticklabels(seq2)
    ax.set_yticklabels(seq1)

    ax.set_xlabel("Sequence 2", position=(0.5, 1.08))  # Adjust label position
    ax.set_ylabel("Sequence 1")
    ax.set_title("Needleman-Wunsch Alignment Matrix", y=1.12)  # Adjust title position

    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
    plt.show()


def save_results(
        aligned_seq1: str,
        aligned_seq2: str,
        stats: Dict[str, float],
        params: Dict[str, int],
        output_file: str = "alignment_result.txt",
) -> None:
    """
    Save alignment results to a text file.

    The output includes:
    - The aligned sequences
    - Algorithm parameters
    - Alignment statistics
    - Optimal path information

    Args:
        aligned_seq1: First aligned sequence.
        aligned_seq2: Second aligned sequence.
        stats: Dictionary of alignment statistics.
        params: Dictionary of algorithm parameters.
        output_file: Path to save the results (default: "alignment_result.txt").

    Returns:
        None
    """
    with open(output_file, "w") as f:
        f.write("=== Needleman-Wunsch Alignment Results ===\n\n")

        f.write("=== Parameters ===\n")
        f.write(f"Match score: {params['match_score']}\n")
        f.write(f"Mismatch penalty: {params['mismatch_penalty']}\n")
        f.write(f"Gap penalty: {params['gap_penalty']}\n\n")

        f.write("=== Optimal Alignment ===\n")
        # Display alignment in blocks of 100 characters for readability
        block_size = 100
        for i in range(0, len(aligned_seq1), block_size):
            f.write(f"Seq1: {aligned_seq1[i:i + block_size]}\n")
            f.write(f"Seq2: {aligned_seq2[i:i + block_size]}\n\n")

        f.write("\n=== Alignment Statistics ===\n")
        f.write(f"Alignment length: {stats['alignment_length']}\n")
        f.write(f"Matches: {stats['matches']}\n")
        f.write(f"Mismatches: {stats['mismatches']}\n")
        f.write(f"Gaps: {stats['gaps']}\n")
        f.write(f"Identity: {stats['identity']:.2f}%\n")
        f.write(f"Gap percentage: {stats['gap_percentage']:.2f}%\n")
        f.write(f"Alignment score: {stats['score']}\n\n")

        f.write("=== Notes ===\n")
        f.write("- The alignment shows one optimal pathway\n")
        f.write("- Gaps are represented by '-' characters\n")
        f.write("- Match/mismatch counts exclude positions with gaps\n")


def main():
    """Main function to run the Needleman-Wunsch alignment."""
    print("=== Needleman-Wunsch Sequence Alignment ===")

    # Input sequences
    input_type = input("Load sequences from (manual/fasta): ").strip().lower()
    if input_type == "fasta":
        file_path = input("Enter FASTA file path: ").strip()
        try:
            seq1, seq2 = read_fasta(file_path)
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}")
            return
    else:
        seq1 = input("Enter first sequence: ").strip().upper()
        seq2 = input("Enter second sequence: ").strip().upper()
        if not (seq1.isalpha() and seq2.isalpha()):
            print("Error: Sequences must contain only letters.")
            return

    # Input parameters
    try:
        match_score = int(input("Match score (default 1): ") or 1)
        mismatch_penalty = int(input("Mismatch penalty (default -1): ") or -1)
        gap_penalty = int(input("Gap penalty (default -2): ") or -2)
    except ValueError:
        print("Error: Parameters must be integers.")
        return

    # Run alignment
    matrix = initialize_matrix(seq1, seq2, gap_penalty)
    matrix = fill_matrix(seq1, seq2, matrix, match_score, mismatch_penalty, gap_penalty)
    aligned_seq1, aligned_seq2, path = traceback(seq1, seq2, matrix, match_score, mismatch_penalty, gap_penalty)
    stats = calculate_statistics(aligned_seq1, aligned_seq2, matrix, match_score, mismatch_penalty, gap_penalty)

    params = {
        "match_score": match_score,
        "mismatch_penalty": mismatch_penalty,
        "gap_penalty": gap_penalty,
    }

    # Display results
    print("\n=== Optimal Alignment ===")
    print(aligned_seq1)
    print(aligned_seq2)
    print("\n=== Statistics ===")
    print(f"Alignment length: {stats['alignment_length']}")
    print(f"Matches: {stats['matches']}")
    print(f"Mismatches: {stats['mismatches']}")
    print(f"Gaps: {stats['gaps']}")
    print(f"Identity: {stats['identity']:.2f}%")
    print(f"Gap percentage: {stats['gap_percentage']:.2f}%")
    print(f"Alignment score: {stats['score']}")

    # Save and plot
    save_results(aligned_seq1, aligned_seq2, stats, params)
    plot_matrix(matrix, seq1, seq2, path)


if __name__ == "__main__":
    main()