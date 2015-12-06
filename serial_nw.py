# Edit distance = 1 for insertion, deletion, and replacement
import numpy as np
import string
def edit_distance(strA, strB, backtrace=False):
    lenA, lenB = len(strA), len(strB)
    if strA and strB:
        # DP table
        # DP[i,j] = edit distance between strA[:(i+1)] and strB[:(j+1)]
        dp = np.zeros((lenA+1, lenB+1), dtype=np.uint)
        # Initialize 
        dp[:,0] = np.arange(lenA+1, dtype=np.uint)
        dp[0,:] = np.arange(lenB+1, dtype=np.uint)
        # Update
        for i in xrange(1, lenA+1):
            for j in xrange(1, lenB+1):
                dp[i,j] = min(dp[i-1,j]+1,
                              dp[i,j-1]+1,
                              dp[i-1,j-1] + (0 if strA[i-1] == strB[j-1] else 1))
        if not backtrace:
            return dp[lenA, lenB]
        else:
            # if backtrace returns edit distance, and aligned sequences and associated backtrace
            # aligned sequences: '-' indicates insertion/deletion
            # backtrace: M=match, D=deletion, I=insertion, S=substitution
            n, m = lenA, lenB
            alignedA = []
            alignedB = []
            backtrace = [] 
            while n > 0 or m > 0:
                minval = dp[n, m]
                prev = np.argmin((dp[n-1,m-1], dp[n-1,m], dp[n,m-1]))
                if prev == 0:
                    # match or substitution
                    alignedA.append(strA[n-1])
                    alignedB.append(strB[m-1])
                    backtrace.append('M' if dp[n-1,m-1] == dp[n,m] else 'S')
                    m -= 1
                    n -= 1
                elif prev == 1:
                    # deletion
                    alignedA.append(strA[n-1])
                    alignedB.append('-')
                    backtrace.append('D')
                    n -= 1
                elif prev == 2:
                    # insertion
                    alignedA.append('-')
                    alignedB.append(strB[m-1])
                    backtrace.append('I')
                    m -= 1
            return dp[lenA, lenB], ''.join(alignedA[::-1]), ''.join(alignedB[::-1]), ''.join(backtrace[::-1])

if __name__ == '__main__':
    strA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    strB = 'CDEFGHIJKLOMNPQRSTUVWXYZCB'
    print 'Seq A    :', strA
    print 'Seq B    :', strB
    res = edit_distance(strA, strB, True)
    print 'Aligned A:', res[1]
    print 'Aligned B:', res[2]
    print 'Backtrace:', res[3]
    print 'Edit Dist:', res[0]


