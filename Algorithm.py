class DPMatrix:
    def __init__(self, sequence):
        self.sequence = sequence
        self.seq_len = len(sequence)
        self.dp_table = [[0 for x in range(self.seq_len)] for x in range(self.seq_len)]

    def rna_folding(self):
        for j in range(self.seq_len-1):
            for i in range(self.seq_len-1):
                j += 1
                if j < self.seq_len:
                    print('i = ', i)
                    print('j = ', j)
                    val1 = 0
                    if self.base_pair(i, j):
                        val1 = self.dp_table[i+1][j-1] + 1
                    print('val1: ', val1)
                    val2 = 0
                    for k in range(i, j):
                        temp_val = self.dp_table[i][k] + self.dp_table[k + 1][j]
                        if temp_val > val2:
                            val2 = temp_val
                    print('val2: ', val2)
                    max_val = max(val1, val2)
                    self.dp_table[i][j] = max_val
        print(self.dp_table)
        return self.dp_table[0][self.seq_len-1]

    def get_matrix(self):
        return self.dp_table

    def base_pair(self, i, j):
        a = (self.sequence[i] == 'A' and self.sequence[j] == 'U')
        b = (self.sequence[i] == 'U' and self.sequence[j] == 'A')
        c = (self.sequence[i] == 'C' and self.sequence[j] == 'G')
        d = (self.sequence[i] == 'G' and self.sequence[j] == 'C')
        return a or b or c or d

    def get_rna_structure(self):
        stack = []
        pairs = []
        stack.append([0, self.seq_len-1])
        while len(stack) > 0:
            curr_pair = stack.pop()
            i = curr_pair[0]
            j = curr_pair[1]
            if i >= j:
                continue
            elif self.dp_table[i+1][j] == self.dp_table[i][j]:
                stack.append([i+1, j])
            elif self.dp_table[i][j-1] == self.dp_table[i][j]:
                stack.append([i, j-1])
            elif self.dp_table[i+1][j-1]+1 == self.dp_table[i][j] and self.base_pair(i, j):
                stack.append([i+1, j-1])
                pairs.append([i, j])
            else:
                for k in range(i+1, j-1):
                    if self.dp_table[i][k]+self.dp_table[k+1][j] == self.dp_table[i][j]:
                        stack.append([i, k])
                        stack.append([k+1, j])
                        break
        return pairs



#seq = "AGCUAU"
seq = "GGGAAAUCC"
matrix = DPMatrix(seq)
ans = matrix.rna_folding()
print(ans)
rna = matrix.get_rna_structure()
print(rna)
