import random
import time
import matplotlib.pyplot as plt

# =====================================================
# ALGORITHM IMPLEMENTATIONS
# =====================================================

# 1) Counting Sort
def counting_sort(array):
    if not array:
        return array
    output = [0] * len(array)
    max_val = max(array)
    count = [0] * (max_val + 1)
    for num in array:
        count[num] += 1
    for i in range(1, len(count)):
        count[i] += count[i - 1]
    for num in reversed(array):
        output[count[num] - 1] = num
        count[num] -= 1
    return output

# 2) Radix Sort
def radix_sort(arr):
    def counting_sort_digit(a, place):
        n = len(a)
        output = [0] * n
        count = [0] * 10

        for num in a:
            index = (num // place) % 10
            count[index] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]
        for i in range(n - 1, -1, -1):
            index = (a[i] // place) % 10
            output[count[index] - 1] = a[i]
            count[index] -= 1
        for i in range(n):
            a[i] = output[i]

    if len(arr) < 2:
        return arr
    max_val = max(arr)
    place = 1
    while max_val // place > 0:
        counting_sort_digit(arr, place)
        place *= 10
    return arr

# 3) Shell Sort
def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    return arr

# 4) Rabin-Karp
def rabin_karp(text, pattern):
    if not pattern or not text:
        return []
    N = len(text)
    M = len(pattern)
    if M > N:
        return []

    p, t = 0, 0
    d = 256
    q = 1000000007
    h = 1
    for _ in range(M - 1):
        h = (h * d) % q

    for i in range(M):
        p = (p * d + ord(pattern[i])) % q
        t = (t * d + ord(text[i])) % q

    indices = []
    for i in range(N - M + 1):
        if p == t:
            # Verify characters
            if text[i:i+M] == pattern:
                indices.append(i)
        if i < N - M:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + M])) % q
            t = (t + q) % q
    return indices

# 5) KMP
def KMP_prefix(s):
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi

def kmp_search(text, pattern):
    if not pattern or not text:
        return []
    N = len(text)
    M = len(pattern)
    if M > N:
        return []

    # Build pi array for pattern
    pi_p = KMP_prefix(pattern)
    matches = []
    j = 0  # index for pattern
    for i in range(N):
        while j > 0 and text[i] != pattern[j]:
            j = pi_p[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == M:
            matches.append(i - M + 1)
            j = pi_p[j - 1]
    return matches

# 6) Trie
class TrieNode:
    def __init__(self, char):
        self.char = char
        self.is_end = False
        self.children = {}

class Trie:
    def __init__(self):
        self.root = TrieNode("")

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode(char)
            node = node.children[char]
        node.is_end = True

    def query(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return True


# =====================================================
# PERFORMANCE COMPARISONS
# =====================================================
def measure_sorting_performance():
    """
    Compare Counting Sort, Radix Sort, and Shell Sort
    by running them on random arrays of increasing size.
    Return a dict: { "CountingSort":[time1,...], "RadixSort":..., "ShellSort":... }
    """
    sort_funcs = {
        "CountingSort": counting_sort,
        "RadixSort":   radix_sort,
        "ShellSort":   shell_sort
    }
    array_sizes = [500, 2000, 5000, 10000]  # You can adjust these
    results = {name: [] for name in sort_funcs}
    
    for n in array_sizes:
        # We'll generate a random array of size n with values in [0..2n]
        # so counting sort doesn't blow up in memory usage.
        arr_original = [random.randint(0, 2*n) for _ in range(n)]
        
        for alg_name, alg_func in sort_funcs.items():
            # Copy array so each algorithm sees the same data
            arr_copy = arr_original[:]
            
            start = time.time()
            alg_func(arr_copy)
            end = time.time()
            
            elapsed = end - start
            results[alg_name].append(elapsed)
    
    # Plot the results
    plt.figure()
    for alg_name in sort_funcs:
        plt.plot(array_sizes, results[alg_name], marker='o', label=alg_name)
    plt.xlabel("Array Size (n)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Sorting Algorithm Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("sorting_performance.png")
    plt.close()

def measure_string_matching_performance():
    """
    Compare Rabin-Karp and KMP string-matching algorithms
    on random strings of increasing size.
    Return a dict: { "RabinKarp":[time1,...], "KMP":[time1,...] }
    """
    match_funcs = {
        "RabinKarp": rabin_karp,
        "KMP":       kmp_search
    }
    text_sizes = [1000, 3000, 6000, 10000]  # adjust as you wish
    results = {name: [] for name in match_funcs}
    
    pattern_length = 10  # we’ll keep pattern fixed length
    
    for n in text_sizes:
        # Generate random text of length n (only lowercase letters)
        text = "".join(chr(random.randint(ord('a'), ord('z'))) for _ in range(n))
        # Generate random pattern of length pattern_length
        pattern = "".join(chr(random.randint(ord('a'), ord('z'))) for _ in range(pattern_length))
        
        for alg_name, alg_func in match_funcs.items():
            start = time.time()
            _ = alg_func(text, pattern)
            end = time.time()
            elapsed = end - start
            results[alg_name].append(elapsed)
    
    # Plot the results
    plt.figure()
    for alg_name in match_funcs:
        plt.plot(text_sizes, results[alg_name], marker='o', label=alg_name)
    plt.xlabel("Text Size (characters)")
    plt.ylabel("Runtime (seconds)")
    plt.title("String-Matching Algorithm Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("string_matching_performance.png")
    plt.close()

# =====================================================
# MAIN DEMO
# =====================================================
if __name__ == "__main__":
    print("Measuring Sorting Performance (Counting, Radix, Shell). Please wait...")
    measure_sorting_performance()
    print(" -> Generated 'sorting_performance.png'")

    print("Measuring String Matching Performance (Rabin-Karp, KMP). Please wait...")
    measure_string_matching_performance()
    print(" -> Generated 'string_matching_performance.png'")

    print("\nDone! Check the two .png files for comparative performance plots.\n")

    # Small demonstration of Trie usage (not timed):
    trie_example = Trie()
    trie_example.insert("algorisfunalgoisgreat")
    print("Trie query('algo') =>", trie_example.query("algo"))
    print("Trie query('fun') =>", trie_example.query("fun"))

import matplotlib.pyplot as plt

# --------------------------------------------------------------
# 1) Counting Sort
# --------------------------------------------------------------
def counting_sort(array):
    """
    Sorts a list of integers using the Counting Sort algorithm.
    """
    output = [0] * len(array)
    count = [0] * (max(array) + 1)

    # Store the count of each element
    for i in array:
        count[i] += 1

    # Calculate the cumulative count
    for j in range(1, len(count)):
        count[j] += count[j - 1]

    # Place the elements in the output array
    for k in range(len(array) - 1, -1, -1):
        output[count[array[k]] - 1] = array[k]
        count[array[k]] -= 1

    return output

# --------------------------------------------------------------
# 2) Radix Sort
# --------------------------------------------------------------
def radix_sort(arr):
    """
    Sorts a list of integers using the Radix Sort algorithm.
    """

    def mod_counting_sort(a, d):
        size = len(a)
        output = [0] * size
        count = [0] * 10

        # Count the occurrences of each digit (according to place d)
        for i in range(size):
            index = a[i] // d
            count[index % 10] += 1

        # Cumulative count
        for j in range(1, 10):
            count[j] += count[j - 1]

        # Build the output array
        for i in range(size - 1, -1, -1):
            index = a[i] // d
            output[count[index % 10] - 1] = a[i]
            count[index % 10] -= 1

        # Copy the sorted elements for this digit back to a
        for f in range(size):
            a[f] = output[f]

    # Main Radix Sort
    maximum = max(arr)
    place = 1
    while maximum // place > 0:
        mod_counting_sort(arr, place)
        place *= 10

    return arr

# --------------------------------------------------------------
# 3) Shell Sort
# --------------------------------------------------------------
def shell_sort(arr):
    """
    Sorts a list of integers using the Shell Sort algorithm.
    """
    gap = len(arr) // 2

    while gap > 0:
        i = 0
        j = gap

        # Compare elements that are gap apart
        while j < len(arr):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
            i += 1
            j += 1

            # Look back from index i to the left
            k = i
            while k - gap >= 0:
                if arr[k - gap] > arr[k]:
                    arr[k - gap], arr[k] = arr[k], arr[k - gap]
                k -= 1

        gap //= 2

    return arr

# --------------------------------------------------------------
# 4) Rabin-Karp
# --------------------------------------------------------------
def rabin_karp(text, pattern):
    """
    Searches for 'pattern' in 'text' using the Rabin-Karp algorithm.
    Returns the list of starting indices where the pattern is found.
    Also returns a list of rolling hash values for visualization.
    """
    N = len(text)
    M = len(pattern)

    # Rolling hash values for pattern and text
    p = 0
    t = 0
    d = 256
    q = 1000000007
    h = 1

    # The value of h is "pow(d, M-1) % q"
    for _ in range(M - 1):
        h = (h * d) % q

    # Calculate the hash value of pattern and first window of text
    for i in range(M):
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q

    indices_found = []
    # We'll keep track of the rolling hash 't' in a list for plotting
    rolling_hash_list = [t]

    # Slide the pattern over text one by one
    for i in range(N - M + 1):
        if p == t:
            # Check for characters one by one
            match = True
            for j in range(M):
                if text[i + j] != pattern[j]:
                    match = False
                    break
            if match:
                indices_found.append(i)

        if i < N - M:
            # Recalculate hash for next window
            t = (d * (t - ord(text[i]) * h) + ord(text[i + M])) % q
            # We might get negative values of t, convert it to positive
            t = (t + q) % q
            rolling_hash_list.append(t)

    return indices_found, rolling_hash_list

# --------------------------------------------------------------
# 5) KMP Algorithm
# --------------------------------------------------------------
def KMP_prefix(s):
    """
    Constructs the prefix (pi) array used in the KMP pattern-matching algorithm.
    Returns the pi array for string s.
    """
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi

def solve_kmp(text, pattern):
    """
    Uses the KMP algorithm to find occurrences of 'pattern' in 'text'.
    Returns a list of starting indices where pattern is found.
    Also returns the prefix array (pi) of pattern+'!'+text for plotting.
    """
    tmp = pattern + '!' + text
    pi = KMP_prefix(tmp)

    m = len(pattern)
    indices_found = []

    # Check from the part corresponding to 'text' in tmp
    for i in range(m + 1, len(pi)):
        if pi[i] == m:
            # Calculate the start index of this match in 'text'
            index = i - 2 * m
            indices_found.append(index)

    return indices_found, pi

# --------------------------------------------------------------
# 6) Trie
# --------------------------------------------------------------
class TrieNode:
    """A node in the trie structure."""
    def __init__(self, char):
        self.char = char
        self.is_end = False
        self.children = {}

class Trie:
    """The trie object."""
    def __init__(self):
        self.root = TrieNode("")

    def insert(self, word):
        """
        Insert 'word' into the trie.
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode(char)
            node = node.children[char]
        node.is_end = True

    def query(self, word):
        """
        Returns True if 'word' exists in the trie (by prefix path), False otherwise.
        """
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# --------------------------------------------------------------
# Generating Charts / Demonstrations
# --------------------------------------------------------------
if __name__ == "__main__":
    # 1) Counting Sort
    arr = [16, 30, 95, 51, 84, 23, 62, 44]
    sorted_arr_counting = counting_sort(arr[:])
    plt.figure()
    plt.bar(range(len(sorted_arr_counting)), sorted_arr_counting)
    plt.title("Counting Sort Result")
    plt.savefig("counting_sort.png")
    plt.close()

    # 2) Radix Sort
    arr = [16, 30, 95, 51, 84, 23, 62, 44]
    sorted_arr_radix = radix_sort(arr[:])
    plt.figure()
    plt.bar(range(len(sorted_arr_radix)), sorted_arr_radix)
    plt.title("Radix Sort Result")
    plt.savefig("radix_sort.png")
    plt.close()

    # 3) Shell Sort
    arr = [16, 30, 95, 51, 84, 23, 62, 44]
    sorted_arr_shell = shell_sort(arr[:])
    plt.figure()
    plt.bar(range(len(sorted_arr_shell)), sorted_arr_shell)
    plt.title("Shell Sort Result")
    plt.savefig("shell_sort.png")
    plt.close()

    # 4) Rabin-Karp
    text_example = "algorisfunalgoisgreat"
    pattern_rk = "fun"
    indices_rk, hash_list = rabin_karp(text_example, pattern_rk)
    # Plot the rolling hash values for visualization
    plt.figure()
    plt.plot(range(len(hash_list)), hash_list)
    plt.title(f"Rabin-Karp Rolling Hash (pattern='{pattern_rk}')")
    plt.savefig("rabin_karp.png")
    plt.close()

    # 5) KMP
    text_example = "algorisfunalgoisgreat"
    pattern_kmp = "fun"
    indices_kmp, pi_array = solve_kmp(text_example, pattern_kmp)
    # Plot the prefix array
    plt.figure()
    plt.plot(range(len(pi_array)), pi_array)
    plt.title(f"KMP Prefix Array (pattern='{pattern_kmp}')")
    plt.savefig("kmp.png")
    plt.close()

    # 6) Trie
    # We'll insert the entire string "algorisfunalgoisgreat" and see how it goes.
    trie_example = Trie()
    word_to_insert = "algorisfunalgoisgreat"
    trie_example.insert(word_to_insert)
    # For a simple numeric chart, let's plot ASCII values of each character in 'word_to_insert'
    ascii_vals = [ord(c) for c in word_to_insert]
    plt.figure()
    plt.bar(range(len(ascii_vals)), ascii_vals)
    plt.title("Trie Example: ASCII Values of 'algorisfunalgoisgreat'")
    plt.savefig("tries.png")
    plt.close()

    print("All algorithms executed. PNG plots generated:")
    print(" - counting_sort.png")
    print(" - radix_sort.png")
    print(" - shell_sort.png")
    print(" - rabin_karp.png")
    print(" - kmp.png")
    print(" - tries.png")
