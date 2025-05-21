import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class MSAGUI:
    """
    A graphical user interface for performing Multiple Sequence Alignment using the Star Method.

    This application allows users to:
    - Input sequences manually or via FASTA files
    - Configure alignment scoring parameters
    - Visualize aligned sequences with color coding
    - View alignment statistics
    - Save results to file

    The alignment is performed using a modified Needleman-Wunsch algorithm with
    a star-based approach that selects a center sequence and aligns all other
    sequences to it.
    """

    def __init__(self, master):
        """Initialize the MSA GUI application.

        Args:
            master: The root Tkinter window
        """
        self.master = master
        master.title("Multiple Sequence Alignment (Star Method)")
        master.geometry("1200x800")

        # Initialize variables
        self.input_method = tk.StringVar(value="manual")
        self.sequences = []
        self.match_score = tk.IntVar(value=2)
        self.mismatch_score = tk.IntVar(value=-1)
        self.gap_penalty = tk.IntVar(value=-2)

        # Set up matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("Alignment Conservation")
        plt.tight_layout()

        # Build the GUI
        self.create_widgets()

    def create_widgets(self):
        """Create and arrange all GUI components."""
        # Input Frame
        input_frame = tk.LabelFrame(self.master, text="Input Sequences", padx=10, pady=10)
        input_frame.pack(pady=10, padx=10, fill="x")

        tk.Radiobutton(input_frame, text="Manual Input", variable=self.input_method, value="manual",
                       command=self.toggle_input_fields).pack(anchor="w")
        self.manual_input_label = tk.Label(input_frame, text="Enter sequences (one per line):")
        self.manual_input_label.pack(anchor="w")
        self.sequence_text = scrolledtext.ScrolledText(input_frame, width=80, height=8)
        self.sequence_text.pack(pady=5)

        tk.Radiobutton(input_frame, text="From FASTA File", variable=self.input_method, value="fasta",
                       command=self.toggle_input_fields).pack(anchor="w")
        self.fasta_frame = tk.Frame(input_frame)
        self.fasta_frame.pack(anchor="w", fill="x")
        self.fasta_label = tk.Label(self.fasta_frame, text="FASTA File:")
        self.fasta_label.pack(side="left", padx=5)
        self.fasta_path_entry = tk.Entry(self.fasta_frame, width=50)
        self.fasta_path_entry.pack(side="left", expand=True, fill="x")
        self.browse_button = tk.Button(self.fasta_frame, text="Browse", command=self.browse_fasta_file)
        self.browse_button.pack(side="left", padx=5)

        self.toggle_input_fields()

        # Scoring Scheme Frame
        scoring_frame = tk.LabelFrame(self.master, text="Scoring Scheme", padx=10, pady=10)
        scoring_frame.pack(pady=10, padx=10, fill="x")

        tk.Label(scoring_frame, text="Match Score:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        tk.Entry(scoring_frame, textvariable=self.match_score, width=10).grid(row=0, column=1, padx=5, pady=2,
                                                                              sticky="w")

        tk.Label(scoring_frame, text="Mismatch Score:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        tk.Entry(scoring_frame, textvariable=self.mismatch_score, width=10).grid(row=1, column=1, padx=5, pady=2,
                                                                                 sticky="w")

        tk.Label(scoring_frame, text="Gap Penalty:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        tk.Entry(scoring_frame, textvariable=self.gap_penalty, width=10).grid(row=2, column=1, padx=5, pady=2,
                                                                              sticky="w")

        # Control Buttons
        button_frame = tk.Frame(self.master, padx=10, pady=5)
        button_frame.pack(pady=5, padx=10, fill="x")

        tk.Button(button_frame, text="Perform Alignment", command=self.perform_alignment).pack(side="left", padx=5)
        tk.Button(button_frame, text="Save Output", command=self.save_output).pack(side="right", padx=5)

        # Output Frame
        output_frame = tk.LabelFrame(self.master, text="Alignment Results", padx=10, pady=10)
        output_frame.pack(pady=10, padx=10, fill="both", expand=True)

        output_frame.grid_columnconfigure(0, weight=1)
        output_frame.grid_columnconfigure(1, weight=1)
        output_frame.grid_rowconfigure(0, weight=0)
        output_frame.grid_rowconfigure(1, weight=3)
        output_frame.grid_rowconfigure(2, weight=0)
        output_frame.grid_rowconfigure(3, weight=1)

        # Left Column Widgets
        self.alignment_label = tk.Label(output_frame, text="Aligned Sequences:")
        self.alignment_label.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.alignment_text = scrolledtext.ScrolledText(output_frame, wrap="none", width=90, height=10)
        self.alignment_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.statistics_label = tk.Label(output_frame, text="Statistics:")
        self.statistics_label.grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.statistics_text = scrolledtext.ScrolledText(output_frame, width=90, height=5)
        self.statistics_text.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)

        # Configure text tags for colored output
        self.alignment_text.tag_configure("match", foreground="green")
        self.alignment_text.tag_configure("mismatch", foreground="red")
        self.alignment_text.tag_configure("gap", foreground="blue")

        # Right Column Widget (Matplotlib Plot)
        self.plot_frame = tk.Frame(output_frame, bd=2, relief="groove")
        self.plot_frame.grid(row=0, column=1, rowspan=4, sticky="nsew", padx=5, pady=5)
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)

        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_plot_widget = self.canvas_plot.get_tk_widget()
        self.canvas_plot_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas_plot, self.plot_frame)
        self.toolbar.update()
        self.canvas_plot_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def toggle_input_fields(self):
        """Toggle between manual input and FASTA file input fields."""
        if self.input_method.get() == "manual":
            self.manual_input_label.pack(anchor="w")
            self.sequence_text.pack(pady=5)
            self.fasta_frame.pack_forget()
        else:
            self.manual_input_label.pack_forget()
            self.sequence_text.pack_forget()
            self.fasta_frame.pack(anchor="w", fill="x")

    def browse_fasta_file(self):
        """Open file dialog to select FASTA file and update path entry."""
        filepath = filedialog.askopenfilename(filetypes=[("FASTA files", "*.fasta *.fa"), ("All files", "*.*")])
        if filepath:
            self.fasta_path_entry.delete(0, tk.END)
            self.fasta_path_entry.insert(0, filepath)

    def read_fasta_file(self, filepath):
        """Read sequences from a FASTA file.

        Args:
            filepath: Path to the FASTA file

        Returns:
            List of sequence strings
        """
        sequences = []
        current_sequence = ""
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('>'):
                        if current_sequence:
                            sequences.append(current_sequence)
                        current_sequence = ""
                    else:
                        current_sequence += line
                if current_sequence:
                    sequences.append(current_sequence)
        except Exception as e:
            messagebox.showerror("File Error", f"Could not read FASTA file: {e}")
            return []
        return sequences

    def get_sequences(self):
        """Get sequences based on selected input method.

        Returns:
            List of sequence strings
        """
        if self.input_method.get() == "manual":
            sequences_str = self.sequence_text.get("1.0", tk.END).strip()
            if not sequences_str:
                messagebox.showwarning("Input Error", "Please enter sequences manually.")
                return []
            return [s.strip().upper() for s in sequences_str.split('\n') if s.strip()]
        else:
            filepath = self.fasta_path_entry.get()
            if not filepath:
                messagebox.showwarning("Input Error", "Please select a FASTA file.")
                return []
            return self.read_fasta_file(filepath)

    def perform_alignment(self):
        """Perform multiple sequence alignment using star method."""
        self.sequences = self.get_sequences()
        if not self.sequences:
            return

        if len(self.sequences) < 2:
            messagebox.showwarning("Input Error", "Please provide at least two sequences for alignment.")
            return

        match_s = self.match_score.get()
        mismatch_s = self.mismatch_score.get()
        gap_p = self.gap_penalty.get()

        try:
            self.alignment_text.delete("1.0", tk.END)
            self.statistics_text.delete("1.0", tk.END)
            self.alignment_text.insert(tk.END, "Performing Star Alignment...\n")

            # Select center sequence (longest)
            center_sequence_index = 0
            max_len = 0
            for i, seq in enumerate(self.sequences):
                if len(seq) > max_len:
                    max_len = len(seq)
                    center_sequence_index = i
            center_sequence = self.sequences[center_sequence_index]
            self.alignment_text.insert(tk.END, f"Center Sequence: {center_sequence}\n\n")

            # Perform pairwise alignments against center sequence
            pairwise_aligned_segments_for_center = []
            for i, seq in enumerate(self.sequences):
                if i == center_sequence_index:
                    continue
                aligned_center_pw, aligned_other_pw = self.needleman_wunsch(
                    center_sequence, seq, match_s, mismatch_s, gap_p
                )
                pairwise_aligned_segments_for_center.append((aligned_center_pw, aligned_other_pw))

            # Build master gapped center sequence
            master_gapped_center_final = list(center_sequence)
            all_center_gap_positions = set()

            for aligned_center_pw, _ in pairwise_aligned_segments_for_center:
                original_center_ptr = 0
                for char_aligned in aligned_center_pw:
                    if char_aligned == '-':
                        all_center_gap_positions.add(original_center_ptr)
                    else:
                        original_center_ptr += 1

            sorted_gap_positions = sorted(list(all_center_gap_positions), reverse=True)
            for pos in sorted_gap_positions:
                master_gapped_center_final.insert(pos, '-')

            # Align all sequences to master gapped center
            final_msa_sequences_str = []
            for i, original_seq in enumerate(self.sequences):
                if i == center_sequence_index:
                    final_msa_sequences_str.append("".join(master_gapped_center_final))
                else:
                    _, aligned_other_final = self.needleman_wunsch(
                        "".join(master_gapped_center_final), original_seq, match_s, mismatch_s, gap_p
                    )
                    final_msa_sequences_str.append(aligned_other_final)

            # Display results
            self.display_alignment(final_msa_sequences_str)
            self.display_matplotlib_graph(final_msa_sequences_str)
            self.calculate_and_display_statistics(final_msa_sequences_str, match_s, mismatch_s, gap_p)

        except Exception as e:
            messagebox.showerror("Alignment Error", f"An error occurred during alignment: {e}")

    def needleman_wunsch(self, seq1, seq2, match_score, mismatch_score, gap_penalty):
        """Perform Needleman-Wunsch global alignment.

        Args:
            seq1: First sequence
            seq2: Second sequence
            match_score: Score for matching characters
            mismatch_score: Score for mismatching characters
            gap_penalty: Penalty for gaps

        Returns:
            Tuple of (aligned_seq1, aligned_seq2)
        """
        n = len(seq1)
        m = len(seq2)

        # Initialize matrices
        score_matrix = [[0] * (m + 1) for _ in range(n + 1)]
        traceback_matrix = [[0] * (m + 1) for _ in range(n + 1)]

        # Initialize first row and column
        for i in range(1, n + 1):
            score_matrix[i][0] = i * gap_penalty
            traceback_matrix[i][0] = 1
        for j in range(1, m + 1):
            score_matrix[0][j] = j * gap_penalty
            traceback_matrix[0][j] = 2

        # Fill matrices
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match = score_matrix[i - 1][j - 1] + (match_score if seq1[i - 1] == seq2[j - 1] else mismatch_score)
                delete = score_matrix[i - 1][j] + gap_penalty
                insert = score_matrix[i][j - 1] + gap_penalty

                score_matrix[i][j] = max(match, delete, insert)

                if score_matrix[i][j] == match:
                    traceback_matrix[i][j] = 0
                elif score_matrix[i][j] == delete:
                    traceback_matrix[i][j] = 1
                else:
                    traceback_matrix[i][j] = 2

        # Traceback
        aligned_seq1 = []
        aligned_seq2 = []
        i, j = n, m

        while i > 0 or j > 0:
            if traceback_matrix[i][j] == 0:
                aligned_seq1.append(seq1[i - 1])
                aligned_seq2.append(seq2[j - 1])
                i -= 1
                j -= 1
            elif traceback_matrix[i][j] == 1:
                aligned_seq1.append(seq1[i - 1])
                aligned_seq2.append('-')
                i -= 1
            else:
                aligned_seq1.append('-')
                aligned_seq2.append(seq2[j - 1])
                j -= 1

        return "".join(aligned_seq1[::-1]), "".join(aligned_seq2[::-1])

    def display_alignment(self, aligned_sequences):
        """Display aligned sequences with color coding.

        Args:
            aligned_sequences: List of aligned sequence strings
        """
        self.alignment_text.delete("1.0", tk.END)
        if not aligned_sequences:
            self.alignment_text.insert(tk.END, "No alignment to display.\n")
            return

        # Calculate consensus line
        consensus_line = []
        max_len = max(len(s) for s in aligned_sequences)

        for col in range(max_len):
            chars_at_col = [seq[col] if col < len(seq) else '-' for seq in aligned_sequences]
            first_char = chars_at_col[0]
            if all(char == first_char for char in chars_at_col) and first_char != '-':
                consensus_line.append(first_char)
            elif all(char == '-' for char in chars_at_col):
                consensus_line.append(' ')
            else:
                consensus_line.append('.')

        consensus_str = "".join(consensus_line)

        # Display sequences with color coding
        for seq_idx, seq in enumerate(aligned_sequences):
            self.alignment_text.insert(tk.END, f"Seq {seq_idx + 1}: ")
            for i, char in enumerate(seq):
                if char == '-':
                    self.alignment_text.insert(tk.END, char, "gap")
                elif i < len(consensus_str) and consensus_str[i] == char and consensus_str[i] != '.':
                    self.alignment_text.insert(tk.END, char, "match")
                else:
                    self.alignment_text.insert(tk.END, char, "mismatch")
            self.alignment_text.insert(tk.END, "\n")

    def display_matplotlib_graph(self, aligned_sequences):
        """Display alignment visualization using matplotlib.

        Args:
            aligned_sequences: List of aligned sequence strings
        """
        self.ax.clear()
        self.ax.set_title("Aligned Sequences View")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        if not aligned_sequences:
            self.ax.text(0.5, 0.5, "No alignment to display",
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes)
            self.canvas_plot.draw()
            return

        max_len = len(aligned_sequences[0]) if aligned_sequences else 0
        num_sequences = len(aligned_sequences)

        if max_len == 0 or num_sequences == 0:
            self.ax.text(0.5, 0.5, "No alignment to display",
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes)
            self.canvas_plot.draw()
            return

        # Color mapping
        color_map = {
            'match': 'green',
            'mismatch': 'red',
            'gap': 'blue'
        }

        # Determine column conservation types
        column_conservation_types = []
        for col in range(max_len):
            column_chars = [seq[col] for seq in aligned_sequences]
            non_gap_chars = [c for c in column_chars if c != '-']
            num_gaps_in_col = column_chars.count('-')

            if num_gaps_in_col == len(column_chars):
                column_conservation_types.append('gap')
            elif len(set(non_gap_chars)) == 1 and non_gap_chars:
                column_conservation_types.append('match')
            else:
                column_conservation_types.append('mismatch')

        # Plot each character
        for row_idx, seq in enumerate(aligned_sequences):
            self.ax.text(-0.7, num_sequences - 1 - row_idx, f"Seq {row_idx + 1}:",
                         ha='right', va='center', fontsize='small', color='black')
            for col_idx, char in enumerate(seq):
                conservation_type = column_conservation_types[col_idx]
                char_color = color_map[conservation_type]
                self.ax.text(col_idx, num_sequences - 1 - row_idx, char,
                             ha='center', va='center', fontsize='small', color=char_color)

        # Adjust plot limits
        self.ax.set_xlim(-1.5, max_len)
        self.ax.set_ylim(-0.5, num_sequences - 0.5)
        self.ax.set_yticks([])
        self.fig.tight_layout()
        self.canvas_plot.draw()

    def calculate_and_display_statistics(self, aligned_sequences, match_s, mismatch_s, gap_p):
        """Calculate and display alignment statistics.

        Args:
            aligned_sequences: List of aligned sequence strings
            match_s: Match score
            mismatch_s: Mismatch score
            gap_p: Gap penalty
        """
        self.statistics_text.delete("1.0", tk.END)
        if not aligned_sequences:
            self.statistics_text.insert(tk.END, "No statistics to display.\n")
            return

        num_sequences = len(aligned_sequences)
        alignment_length = len(aligned_sequences[0])

        total_gaps = 0
        conserved_columns = 0
        overall_matches = 0
        overall_mismatches = 0

        for col in range(alignment_length):
            column_chars = [seq[col] for seq in aligned_sequences]
            non_gap_chars = [c for c in column_chars if c != '-']
            num_gaps_in_col = column_chars.count('-')

            total_gaps += num_gaps_in_col

            if not non_gap_chars:
                continue

            if len(set(non_gap_chars)) == 1:
                conserved_columns += 1
                overall_matches += 1
            else:
                overall_mismatches += 1

        identity_percent = (conserved_columns / alignment_length) * 100 if alignment_length > 0 else 0

        self.statistics_text.insert(tk.END, f"Program Parameters:\n")
        self.statistics_text.insert(tk.END, f"  Match Score: {match_s}\n")
        self.statistics_text.insert(tk.END, f"  Mismatch Score: {mismatch_s}\n")
        self.statistics_text.insert(tk.END, f"  Gap Penalty: {gap_p}\n\n")

        self.statistics_text.insert(tk.END, f"Alignment Statistics:\n")
        self.statistics_text.insert(tk.END, f"  Alignment Length: {alignment_length}\n")
        self.statistics_text.insert(tk.END, f"  Conserved Columns: {conserved_columns}\n")
        self.statistics_text.insert(tk.END, f"  Overall Identity Percentage: {identity_percent:.2f}%\n")
        self.statistics_text.insert(tk.END, f"  Total Matches (approx): {overall_matches}\n")
        self.statistics_text.insert(tk.END, f"  Total Mismatches (approx): {overall_mismatches}\n")
        self.statistics_text.insert(tk.END, f"  Total Gaps (approx, sum of all gaps): {total_gaps}\n")

    def save_output(self):
        """Save alignment results to a text file."""
        if not self.alignment_text.get("1.0", tk.END).strip():
            messagebox.showwarning("Save Error", "No alignment results to save.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    f.write("--- Multiple Sequence Alignment Results ---\n\n")
                    f.write("Program Parameters:\n")
                    f.write(f"  Match Score: {self.match_score.get()}\n")
                    f.write(f"  Mismatch Score: {self.mismatch_score.get()}\n")
                    f.write(f"  Gap Penalty: {self.gap_penalty.get()}\n\n")
                    f.write("Aligned Sequences:\n")
                    f.write(self.alignment_text.get("1.0", tk.END).strip() + "\n\n")
                    f.write("Statistics:\n")
                    f.write(self.statistics_text.get("1.0", tk.END).strip() + "\n")
                messagebox.showinfo("Save Successful", "Alignment results saved successfully!")
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save file: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MSAGUI(root)
    root.mainloop()