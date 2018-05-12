class DPMatrix:
    def __init__(self, sequence):
        self.sequence = sequence
        self.dp_table = [[0 for x in range(len(sequence))] for x in range(len(sequence))]

    def rna_folding(self):
        seq_len = len(self.sequence)
        print(self.dp_table)
        it = 1
        for j in range(seq_len-1):
            for i in range(seq_len-1):
                j += 1
                if j < seq_len:
                    print('i = ', i)
                    print('j = ', j)
                    a = (self.sequence[i] == 'A' and self.sequence[j] == 'U')
                    b = (self.sequence[i] == 'U' and self.sequence[j] == 'A')
                    c = (self.sequence[i] == 'C' and self.sequence[j] == 'G')
                    d = (self.sequence[i] == 'G' and self.sequence[j] == 'C')
                    val1 = 0
                    print('a: ', a, ' b: ', b, ' c: ', c, ' d :', d)
                    if a or b or c or d:
                        val1 = self.dp_table[i + 1][j - 1] + 1
                    print('val1: ', val1)
                    val2 = 0
                    for k in range(i, j):
                        temp_val = self.dp_table[i][k] + self.dp_table[k + 1][j]
                        if temp_val > val2:
                            val2 = temp_val
                    print('val2: ', val2)
                    max_val = max(val1, val2)
                    self.dp_table[i][j] = max_val
                it += 1
        print(self.dp_table)
        return self.dp_table[0][seq_len-1]

    def get_matrix(self):
        return self.dp_table

seq = "AGCUAU"
matrix = DPMatrix(seq)
ans = matrix.rna_folding()
print(ans)
