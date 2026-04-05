# Blind 75 + extras (108 problems total)
# Original 75 + 33 missing Blind 75 problems (Graphs, DP, Greedy, Intervals, Math, Bit Manipulation)

PROBLEMS = [

    # ─────────────────────────────────────────────────────────────
    # ARRAYS & HASHING
    # ─────────────────────────────────────────────────────────────
    {
        "id": 1,
        "title": "Two Sum",
        "category": "Arrays & Hashing",
        "difficulty": "Easy",
        "description": """\
Given an array of integers `nums` and an integer `target`, return the **indices** of the two numbers that add up to `target`.

You may assume that each input would have **exactly one solution**, and you may not use the same element twice.

**Example 1:**
```
Input:  nums = [2, 7, 11, 15], target = 9
Output: [0, 1]   # nums[0] + nums[1] = 2 + 7 = 9
```

**Example 2:**
```
Input:  nums = [3, 2, 4], target = 6
Output: [1, 2]
```

**Constraints:**
- 2 ≤ len(nums) ≤ 10⁴
- -10⁹ ≤ nums[i] ≤ 10⁹
- Only one valid answer exists.
""",
        "python_tips": """\
**Key Python concept: Dictionaries (hash maps)**

A Python `dict` lets you look up values in O(1) time — much faster than scanning the whole list again.

**The core idea:**
- As you walk through the list, ask: "Have I already seen the number I need to pair with this one?"
- If `target = 9` and current number is `2`, you need `9 - 2 = 7`. Check if `7` is in your dict.
- If yes → you found the pair! Return both indices.
- If no → store `{2: 0}` (value → index) and move on.

**Useful Python syntax to know:**
```python3
d = {}          # empty dictionary
d[key] = val    # store a value
key in d        # check if key exists (returns True/False)
d[key]          # retrieve value
```

**Time complexity:** O(n) — one pass through the list.
**Space complexity:** O(n) — at most n entries in the dict.
""",
        "starter_code": """\
def two_sum(nums, target):
    \"\"\"
    Args:
        nums   (list[int]): list of integers
        target (int):       target sum

    Returns:
        list[int]: indices [i, j] such that nums[i] + nums[j] == target
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([2, 7, 11, 15], 9),  "expected": [0, 1]},
            {"input": ([3, 2, 4],      6),  "expected": [1, 2]},
            {"input": ([3, 3],         6),  "expected": [0, 1]},
        ],
        "solution": """\
def two_sum(nums, target):
    seen = {}            # maps value -> index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
""",
    },

    {
        "id": 2,
        "title": "Best Time to Buy and Sell Stock",
        "category": "Arrays & Hashing",
        "difficulty": "Easy",
        "description": """\
You are given an array `prices` where `prices[i]` is the price of a stock on day `i`.

You want to **buy on one day** and **sell on a later day** to maximize profit.
Return the **maximum profit**. If no profit is possible, return `0`.

**Example 1:**
```
Input:  prices = [7, 1, 5, 3, 6, 4]
Output: 5   # buy at 1, sell at 6
```

**Example 2:**
```
Input:  prices = [7, 6, 4, 3, 1]
Output: 0   # prices only go down
```
""",
        "python_tips": """\
**Key Python concept: Tracking minimum with a variable**

Use two variables as you scan left to right:
- `min_price` — the lowest price seen so far (best day to buy)
- `max_profit` — the best profit seen so far

For each price, check: `price - min_price`. If it beats `max_profit`, update it.

**Useful Python built-ins:**
```python3
min(a, b)   # returns the smaller of two values
max(a, b)   # returns the larger
float('inf')  # represents positive infinity (good initial "min" value)
```

**Time complexity:** O(n) — one pass.
**Space complexity:** O(1) — only two extra variables.
""",
        "starter_code": """\
def max_profit(prices):
    \"\"\"
    Args:
        prices (list[int]): stock prices by day

    Returns:
        int: maximum profit possible (0 if none)
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([7, 1, 5, 3, 6, 4],), "expected": 5},
            {"input": ([7, 6, 4, 3, 1],),     "expected": 0},
            {"input": ([1, 2],),               "expected": 1},
        ],
        "solution": """\
def max_profit(prices):
    min_price  = float('inf')
    max_profit = 0
    for price in prices:
        min_price  = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    return max_profit
""",
    },

    {
        "id": 3,
        "title": "Contains Duplicate",
        "category": "Arrays & Hashing",
        "difficulty": "Easy",
        "description": """\
Given an integer array `nums`, return `True` if any value appears **more than once**, or `False` if every element is distinct.

**Example 1:**
```
Input:  nums = [1, 2, 3, 1]
Output: True
```

**Example 2:**
```
Input:  nums = [1, 2, 3, 4]
Output: False
```
""",
        "python_tips": """\
**Key Python concept: Sets**

A Python `set` stores only **unique** values. If you add a duplicate, it is ignored.

**Two approaches:**
1. Compare `len(nums)` to `len(set(nums))` — if different, there's a duplicate.
2. Walk through `nums`, and for each element check if it's already in a set; if so return `True`.

```python3
s = set()
s.add(x)      # add element x
x in s        # check membership in O(1)
```

**Time complexity:** O(n)
**Space complexity:** O(n)
""",
        "starter_code": """\
def contains_duplicate(nums):
    \"\"\"
    Args:
        nums (list[int])

    Returns:
        bool: True if any duplicate exists
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([1, 2, 3, 1],),    "expected": True},
            {"input": ([1, 2, 3, 4],),    "expected": False},
            {"input": ([1, 1, 1, 3, 3],), "expected": True},
        ],
        "solution": """\
def contains_duplicate(nums):
    return len(nums) != len(set(nums))
""",
    },

    {
        "id": 4,
        "title": "Product of Array Except Self",
        "category": "Arrays & Hashing",
        "difficulty": "Medium",
        "description": """\
Given an integer array `nums`, return an array `output` such that `output[i]` is the product of all elements **except** `nums[i]`.

You must solve it in **O(n)** time and **without using division**.

**Example:**
```
Input:  nums   = [1, 2, 3, 4]
Output: output = [24, 12, 8, 6]
  # output[0] = 2*3*4 = 24
  # output[1] = 1*3*4 = 12
  # output[2] = 1*2*4 = 8
  # output[3] = 1*2*3 = 6
```
""",
        "python_tips": """\
**Key concept: Prefix & Suffix products**

Split the problem into two passes:
1. **Left pass:** `prefix[i]` = product of everything to the LEFT of index i.
2. **Right pass:** `suffix[i]` = product of everything to the RIGHT of index i.
3. The answer at each index is `prefix[i] * suffix[i]`.

You can do this with O(1) extra space (besides output) by using a running variable.

```python3
result = [1] * len(nums)   # initialise output to all 1s
```

Walk left → right filling prefix products, then right → left multiplying in suffix products.
""",
        "starter_code": """\
def product_except_self(nums):
    \"\"\"
    Args:
        nums (list[int])

    Returns:
        list[int]: product of all elements except self
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([1, 2, 3, 4],),     "expected": [24, 12, 8, 6]},
            {"input": ([-1, 1, 0, -3, 3],),"expected": [0, 0, 9, 0, 0]},
        ],
        "solution": """\
def product_except_self(nums):
    n      = len(nums)
    result = [1] * n

    prefix = 1
    for i in range(n):
        result[i] = prefix
        prefix   *= nums[i]

    suffix = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix
        suffix    *= nums[i]

    return result
""",
    },

    {
        "id": 5,
        "title": "Maximum Subarray",
        "category": "Arrays & Hashing",
        "difficulty": "Medium",
        "description": """\
Given an integer array `nums`, find the **contiguous subarray** (at least one element) which has the **largest sum** and return that sum.

**Example 1:**
```
Input:  nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
Output: 6    # subarray [4, -1, 2, 1]
```

**Example 2:**
```
Input:  nums = [1]
Output: 1
```
""",
        "python_tips": """\
**Key algorithm: Kadane's Algorithm**

Walk through the array keeping track of:
- `current_sum` — the best sum ending at the current position
- `max_sum` — the global best seen so far

At each step: if `current_sum` drops below 0, reset it to 0 (it's better to start fresh than drag a negative sum forward).

```python3
current_sum = 0
max_sum     = nums[0]        # handle all-negative arrays

for num in nums:
    current_sum = max(num, current_sum + num)
    max_sum     = max(max_sum, current_sum)
```

**Time:** O(n) | **Space:** O(1)
""",
        "starter_code": """\
def max_subarray(nums):
    \"\"\"
    Args:
        nums (list[int])

    Returns:
        int: largest sum of any contiguous subarray
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([-2, 1, -3, 4, -1, 2, 1, -5, 4],), "expected": 6},
            {"input": ([1],),                               "expected": 1},
            {"input": ([5, 4, -1, 7, 8],),                 "expected": 23},
        ],
        "solution": """\
def max_subarray(nums):
    current_sum = nums[0]
    max_sum     = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum     = max(max_sum, current_sum)
    return max_sum
""",
    },

    # ─────────────────────────────────────────────────────────────
    # TWO POINTERS
    # ─────────────────────────────────────────────────────────────
    {
        "id": 6,
        "title": "Valid Palindrome",
        "category": "Two Pointers",
        "difficulty": "Easy",
        "description": """\
A phrase is a palindrome if, after converting all uppercase letters to lowercase and removing all non-alphanumeric characters, it reads the same forward and backward.

Given a string `s`, return `True` if it is a palindrome, `False` otherwise.

**Example 1:**
```
Input:  s = "A man, a plan, a canal: Panama"
Output: True   # "amanaplanacanalpanama"
```

**Example 2:**
```
Input:  s = "race a car"
Output: False
```
""",
        "python_tips": """\
**Key Python concepts: string methods + two-pointer technique**

**Step 1 — Clean the string:**
```python3
s.isalnum()   # True if character is letter or digit
s.lower()     # convert to lowercase
```

Build a cleaned version, or check characters in-place.

**Step 2 — Two pointers:**
Place one pointer at the start (`left = 0`) and one at the end (`right = len(s)-1`).
Move them toward each other, skipping non-alphanumeric chars, comparing as you go.

If at any point `s[left] != s[right]`, return `False`.
If the pointers cross, return `True`.

**Alternative one-liner (after cleaning):**
```python3
cleaned = [c.lower() for c in s if c.isalnum()]
return cleaned == cleaned[::-1]   # [::-1] reverses a list
```
""",
        "starter_code": """\
def is_palindrome(s):
    \"\"\"
    Args:
        s (str)

    Returns:
        bool: True if s is a palindrome (ignoring case and non-alphanumeric chars)
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ("A man, a plan, a canal: Panama",), "expected": True},
            {"input": ("race a car",),                      "expected": False},
            {"input": (" ",),                               "expected": True},
        ],
        "solution": """\
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left  += 1
        right -= 1
    return True
""",
    },

    {
        "id": 7,
        "title": "3Sum",
        "category": "Two Pointers",
        "difficulty": "Medium",
        "description": """\
Given an integer array `nums`, return all triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, `j != k`, and `nums[i] + nums[j] + nums[k] == 0`.

The solution set must not contain duplicate triplets.

**Example:**
```
Input:  nums = [-1, 0, 1, 2, -1, -4]
Output: [[-1, -1, 2], [-1, 0, 1]]
```
""",
        "python_tips": """\
**Key idea: Sort + Two Pointers**

1. **Sort** the array first. This lets us use two pointers efficiently and skip duplicates.
2. Fix one number `nums[i]` and use two pointers (`left`, `right`) on the rest of the array to find pairs that sum to `-nums[i]`.
3. Skip duplicate values for `i`, `left`, and `right` to avoid repeated triplets.

```python3
nums.sort()
for i in range(len(nums) - 2):
    if i > 0 and nums[i] == nums[i-1]:
        continue    # skip duplicate for i
    left, right = i + 1, len(nums) - 1
    while left < right:
        total = nums[i] + nums[left] + nums[right]
        if total == 0:
            result.append([nums[i], nums[left], nums[right]])
            # skip duplicates for left and right ...
        elif total < 0:
            left  += 1
        else:
            right -= 1
```
""",
        "starter_code": """\
def three_sum(nums):
    \"\"\"
    Args:
        nums (list[int])

    Returns:
        list[list[int]]: all unique triplets summing to 0
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([-1, 0, 1, 2, -1, -4],), "expected": [[-1, -1, 2], [-1, 0, 1]]},
            {"input": ([0, 1, 1],),               "expected": []},
            {"input": ([0, 0, 0],),               "expected": [[0, 0, 0]]},
        ],
        "solution": """\
def three_sum(nums):
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left]  == nums[left  + 1]: left  += 1
                while left < right and nums[right] == nums[right - 1]: right -= 1
                left  += 1
                right -= 1
            elif total < 0:
                left  += 1
            else:
                right -= 1
    return result
""",
    },

    # ─────────────────────────────────────────────────────────────
    # SLIDING WINDOW
    # ─────────────────────────────────────────────────────────────
    {
        "id": 8,
        "title": "Longest Substring Without Repeating Characters",
        "category": "Sliding Window",
        "difficulty": "Medium",
        "description": """\
Given a string `s`, find the length of the **longest substring without repeating characters**.

**Example 1:**
```
Input:  s = "abcabcbb"
Output: 3    # "abc"
```

**Example 2:**
```
Input:  s = "bbbbb"
Output: 1    # "b"
```

**Example 3:**
```
Input:  s = "pwwkew"
Output: 3    # "wke"
```
""",
        "python_tips": """\
**Key technique: Sliding Window with a Set**

Maintain a window `[left, right]` containing no repeated characters.
- Use a `set` to track characters currently in the window.
- Move `right` forward, adding characters.
- If `s[right]` is already in the set, shrink the window from the left until the duplicate is removed.
- Track the maximum window size seen.

```python3
char_set = set()
left = 0
max_len = 0

for right in range(len(s)):
    while s[right] in char_set:
        char_set.remove(s[left])
        left += 1
    char_set.add(s[right])
    max_len = max(max_len, right - left + 1)
```

**Time:** O(n) | **Space:** O(min(n, alphabet_size))
""",
        "starter_code": """\
def length_of_longest_substring(s):
    \"\"\"
    Args:
        s (str)

    Returns:
        int: length of the longest substring without repeating characters
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ("abcabcbb",), "expected": 3},
            {"input": ("bbbbb",),    "expected": 1},
            {"input": ("pwwkew",),   "expected": 3},
            {"input": ("",),         "expected": 0},
        ],
        "solution": """\
def length_of_longest_substring(s):
    char_set = set()
    left     = 0
    max_len  = 0
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)
    return max_len
""",
    },

    # ─────────────────────────────────────────────────────────────
    # BINARY SEARCH
    # ─────────────────────────────────────────────────────────────
    {
        "id": 9,
        "title": "Binary Search",
        "category": "Binary Search",
        "difficulty": "Easy",
        "description": """\
Given a **sorted** array of integers `nums` and a target integer `target`, return the **index** of `target` if found, or `-1` if not.

You must solve it in **O(log n)** time.

**Example 1:**
```
Input:  nums = [-1, 0, 3, 5, 9, 12], target = 9
Output: 4
```

**Example 2:**
```
Input:  nums = [-1, 0, 3, 5, 9, 12], target = 2
Output: -1
```
""",
        "python_tips": """\
**Key algorithm: Binary Search**

Because the array is sorted, you can eliminate half the remaining elements at each step.

Keep track of `left` and `right` boundaries. At each step:
1. Find the middle index: `mid = (left + right) // 2`
2. If `nums[mid] == target` → found it!
3. If `nums[mid] < target`  → target must be in the **right** half → `left = mid + 1`
4. If `nums[mid] > target`  → target must be in the **left** half → `right = mid - 1`

```python3
left, right = 0, len(nums) - 1
while left <= right:
    mid = (left + right) // 2
    ...
```

**Time:** O(log n) | **Space:** O(1)
""",
        "starter_code": """\
def search(nums, target):
    \"\"\"
    Args:
        nums   (list[int]): sorted array
        target (int)

    Returns:
        int: index of target, or -1 if not found
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([-1, 0, 3, 5, 9, 12], 9),  "expected": 4},
            {"input": ([-1, 0, 3, 5, 9, 12], 2),  "expected": -1},
            {"input": ([5], 5),                    "expected": 0},
        ],
        "solution": """\
def search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
""",
    },

    # ─────────────────────────────────────────────────────────────
    # TREES
    # ─────────────────────────────────────────────────────────────
    {
        "id": 10,
        "title": "Invert Binary Tree",
        "category": "Trees",
        "difficulty": "Easy",
        "description": """\
Given the `root` of a binary tree, invert it (mirror it), and return the root.

**Example:**
```
Input:
        4
       / \\
      2   7
     / \\ / \\
    1  3 6  9

Output:
        4
       / \\
      7   2
     / \\ / \\
    9  6 3  1
```
""",
        "python_tips": """\
**Key concept: Tree Node & Recursion**

A binary tree node is defined as:
```python3
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val   = val
        self.left  = left
        self.right = right
```

To invert a tree, simply swap the left and right children — then recursively do the same for both subtrees.

**Recursive approach:**
```python3
def invert_tree(root):
    if root is None:
        return None
    root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root
```

The swap `a, b = b, a` is a Python idiom for swapping two variables without a temporary.

**Time:** O(n) — visits every node once.
""",
        "starter_code": """\
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val   = val
        self.left  = left
        self.right = right

def invert_tree(root):
    \"\"\"
    Args:
        root (TreeNode): root of a binary tree

    Returns:
        TreeNode: root of the inverted tree
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {
                "input": ("tree:[4,2,7,1,3,6,9]",),
                "expected": "tree:[4,7,2,9,6,3,1]",
                "is_tree": True,
            },
            {
                "input": ("tree:[2,1,3]",),
                "expected": "tree:[2,3,1]",
                "is_tree": True,
            },
        ],
        "solution": """\
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val   = val
        self.left  = left
        self.right = right

def invert_tree(root):
    if root is None:
        return None
    root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root
""",
    },

    # ═══════════════════════════════════════════════════════════════
    # STUB PROBLEMS (full title + description + test cases; learn
    # content intentionally minimal — marked for future expansion)
    # ═══════════════════════════════════════════════════════════════

    # ── Arrays & Hashing (stubs) ──────────────────────────────────
    {
        "id": 11,
        "title": "Valid Anagram",
        "category": "Arrays & Hashing",
        "difficulty": "Easy",
        "description": """\
Given two strings `s` and `t`, return `True` if `t` is an anagram of `s`, and `False` otherwise.
An **anagram** is a word formed by rearranging all the letters of another word.

**Example:**
```
Input:  s = "anagram", t = "nagaram"
Output: True
```
""",
        "python_tips": "Hint: use `sorted()` or `collections.Counter` to compare character frequencies.",
        "starter_code": "def is_anagram(s, t):\n    pass\n",
        "test_cases": [
            {"input": ("anagram", "nagaram"), "expected": True},
            {"input": ("rat", "car"),          "expected": False},
        ],
        "solution": """\
def is_anagram(s, t):
    from collections import Counter
    return Counter(s) == Counter(t)
""",
    },
    {
        "id": 12,
        "title": "Group Anagrams",
        "category": "Arrays & Hashing",
        "difficulty": "Medium",
        "description": """\
Given an array of strings `strs`, group the anagrams together.

**Example:**
```
Input:  strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
```
""",
        "python_tips": "Hint: sort each word to get a canonical key, then group by that key using a dict.",
        "starter_code": "def group_anagrams(strs):\n    pass\n",
        "test_cases": [
            {
                "input": (["eat", "tea", "tan", "ate", "nat", "bat"],),
                "expected": [["bat"], ["nat", "tan"], ["ate", "eat", "tea"]],
                "unordered_groups": True,
            },
        ],
        "solution": """\
def group_anagrams(strs):
    from collections import defaultdict
    groups = defaultdict(list)
    for s in strs:
        groups[tuple(sorted(s))].append(s)
    return list(groups.values())
""",
    },
    {
        "id": 13,
        "title": "Top K Frequent Elements",
        "category": "Arrays & Hashing",
        "difficulty": "Medium",
        "description": """\
Given an integer array `nums` and an integer `k`, return the `k` most frequent elements.

**Example:**
```
Input:  nums = [1,1,1,2,2,3], k = 2
Output: [1, 2]
```
""",
        "python_tips": "Hint: use `collections.Counter` then `most_common(k)`.",
        "starter_code": "def top_k_frequent(nums, k):\n    pass\n",
        "test_cases": [
            {"input": ([1, 1, 1, 2, 2, 3], 2), "expected": [1, 2], "unordered": True},
            {"input": ([1], 1),                  "expected": [1]},
        ],
        "solution": """\
def top_k_frequent(nums, k):
    from collections import Counter
    return [x for x, _ in Counter(nums).most_common(k)]
""",
    },
    {
        "id": 14,
        "title": "Encode and Decode Strings",
        "category": "Arrays & Hashing",
        "difficulty": "Medium",
        "description": """\
Design an algorithm to encode a list of strings to a single string, and decode it back.

**Example:**
```
Input:  ["lint","code","love","you"]
Encoded: "4#lint4#code4#love3#you"
Decoded: ["lint","code","love","you"]
```
""",
        "python_tips": "Hint: prefix each string with its length and a delimiter, e.g., `'4#lint'`.",
        "starter_code": "def encode(strs):\n    pass\n\ndef decode(s):\n    pass\n",
        "test_cases": [
            {
                "input": (["lint", "code", "love", "you"],),
                "expected": ["lint", "code", "love", "you"],
                "encode_decode": True,
            },
        ],
        "solution": """\
def encode(strs):
    return ''.join(f'{len(s)}#{s}' for s in strs)

def decode(s):
    result, i = [], 0
    while i < len(s):
        j = s.index('#', i)
        length = int(s[i:j])
        result.append(s[j+1:j+1+length])
        i = j + 1 + length
    return result
""",
    },
    {
        "id": 15,
        "title": "Longest Consecutive Sequence",
        "category": "Arrays & Hashing",
        "difficulty": "Medium",
        "description": """\
Given an unsorted array of integers `nums`, return the length of the longest consecutive elements sequence.
Must run in **O(n)** time.

**Example:**
```
Input:  nums = [100,4,200,1,3,2]
Output: 4    # [1,2,3,4]
```
""",
        "python_tips": "Hint: put all numbers in a set. A sequence starts at `n` only if `n-1` is NOT in the set.",
        "starter_code": "def longest_consecutive(nums):\n    pass\n",
        "test_cases": [
            {"input": ([100, 4, 200, 1, 3, 2],), "expected": 4},
            {"input": ([0, 3, 7, 2, 5, 8, 4, 6, 0, 1],), "expected": 9},
        ],
        "solution": """\
def longest_consecutive(nums):
    num_set = set(nums)
    best = 0
    for n in num_set:
        if n - 1 not in num_set:
            length = 1
            while n + length in num_set:
                length += 1
            best = max(best, length)
    return best
""",
    },

    # ── Two Pointers (stubs) ──────────────────────────────────────
    {
        "id": 16,
        "title": "Container With Most Water",
        "category": "Two Pointers",
        "difficulty": "Medium",
        "description": """\
Given `n` non-negative integers representing heights of vertical lines, find two lines that together with the x-axis form a container that holds the most water.

**Example:**
```
Input:  height = [1,8,6,2,5,4,8,3,7]
Output: 49
```
""",
        "python_tips": "Hint: start with the widest container (left=0, right=end) and move the shorter side inward.",
        "starter_code": "def max_area(height):\n    pass\n",
        "test_cases": [
            {"input": ([1, 8, 6, 2, 5, 4, 8, 3, 7],), "expected": 49},
            {"input": ([1, 1],), "expected": 1},
        ],
        "solution": """\
def max_area(height):
    left, right = 0, len(height) - 1
    best = 0
    while left < right:
        area = min(height[left], height[right]) * (right - left)
        best = max(best, area)
        if height[left] < height[right]:
            left  += 1
        else:
            right -= 1
    return best
""",
    },
    {
        "id": 17,
        "title": "Trapping Rain Water",
        "category": "Two Pointers",
        "difficulty": "Hard",
        "description": """\
Given `n` non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

**Example:**
```
Input:  height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
```
""",
        "python_tips": "Hint: two-pointer approach — track `left_max` and `right_max`, water at index = min(left_max, right_max) - height[i].",
        "starter_code": "def trap(height):\n    pass\n",
        "test_cases": [
            {"input": ([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1],), "expected": 6},
            {"input": ([4, 2, 0, 3, 2, 5],), "expected": 9},
        ],
        "solution": """\
def trap(height):
    if not height: return 0
    left, right = 0, len(height) - 1
    left_max = right_max = water = 0
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    return water
""",
    },

    # ── Sliding Window (stubs) ────────────────────────────────────
    {
        "id": 18,
        "title": "Longest Repeating Character Replacement",
        "category": "Sliding Window",
        "difficulty": "Medium",
        "description": """\
You are given a string `s` and an integer `k`. You can choose any character of the string and change it to any other uppercase English character, at most `k` times. Return the length of the longest substring containing the same letter after performing those operations.

**Example:**
```
Input:  s = "AABABBA", k = 1
Output: 4
```
""",
        "python_tips": "Hint: sliding window. The window is valid when `(window_size - max_freq) <= k`.",
        "starter_code": "def character_replacement(s, k):\n    pass\n",
        "test_cases": [
            {"input": ("ABAB", 2), "expected": 4},
            {"input": ("AABABBA", 1), "expected": 4},
        ],
        "solution": """\
def character_replacement(s, k):
    count = {}
    left = max_freq = result = 0
    for right in range(len(s)):
        count[s[right]] = count.get(s[right], 0) + 1
        max_freq = max(max_freq, count[s[right]])
        while (right - left + 1) - max_freq > k:
            count[s[left]] -= 1
            left += 1
        result = max(result, right - left + 1)
    return result
""",
    },
    {
        "id": 19,
        "title": "Minimum Window Substring",
        "category": "Sliding Window",
        "difficulty": "Hard",
        "description": """\
Given two strings `s` and `t`, return the minimum window substring of `s` that contains all characters of `t`. If no such substring exists, return `""`.

**Example:**
```
Input:  s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
```
""",
        "python_tips": "Hint: sliding window with two frequency counters — one for `t`, one for the current window.",
        "starter_code": "def min_window(s, t):\n    pass\n",
        "test_cases": [
            {"input": ("ADOBECODEBANC", "ABC"), "expected": "BANC"},
            {"input": ("a", "a"), "expected": "a"},
            {"input": ("a", "aa"), "expected": ""},
        ],
        "solution": """\
def min_window(s, t):
    from collections import Counter
    need = Counter(t)
    missing = len(t)
    best = ""
    left = 0
    for right, c in enumerate(s):
        if need[c] > 0:
            missing -= 1
        need[c] -= 1
        if missing == 0:
            while need[s[left]] < 0:
                need[s[left]] += 1
                left += 1
            window = s[left:right+1]
            if not best or len(window) < len(best):
                best = window
            need[s[left]] += 1
            missing += 1
            left += 1
    return best
""",
    },
    {
        "id": 20,
        "title": "Sliding Window Maximum",
        "category": "Sliding Window",
        "difficulty": "Hard",
        "description": """\
Given an array `nums` and a sliding window of size `k`, return the maximum value in each window position.

**Example:**
```
Input:  nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
```
""",
        "python_tips": "Hint: use a monotonic deque (collections.deque) to track the maximum in O(n) total.",
        "starter_code": "def max_sliding_window(nums, k):\n    pass\n",
        "test_cases": [
            {"input": ([1, 3, -1, -3, 5, 3, 6, 7], 3), "expected": [3, 3, 5, 5, 6, 7]},
            {"input": ([1], 1), "expected": [1]},
        ],
        "solution": """\
def max_sliding_window(nums, k):
    from collections import deque
    dq, result = deque(), []
    for i, n in enumerate(nums):
        while dq and nums[dq[-1]] < n:
            dq.pop()
        dq.append(i)
        if dq[0] == i - k:
            dq.popleft()
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result
""",
    },

    # ── Stack ─────────────────────────────────────────────────────
    {
        "id": 21,
        "title": "Valid Parentheses",
        "category": "Stack",
        "difficulty": "Easy",
        "description": """\
Given a string `s` containing just `(`, `)`, `{`, `}`, `[`, `]`, determine if the input string is valid.

**Example:**
```
Input:  s = "()[]{}"
Output: True

Input:  s = "(]"
Output: False
```
""",
        "python_tips": "Hint: use a stack. Push opening brackets; on a closing bracket, pop and check it matches.",
        "starter_code": "def is_valid(s):\n    pass\n",
        "test_cases": [
            {"input": ("()",),      "expected": True},
            {"input": ("()[]{}", ), "expected": True},
            {"input": ("(]",),      "expected": False},
            {"input": ("([)]",),    "expected": False},
        ],
        "solution": """\
def is_valid(s):
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    for c in s:
        if c in pairs:
            if not stack or stack[-1] != pairs[c]:
                return False
            stack.pop()
        else:
            stack.append(c)
    return len(stack) == 0
""",
    },
    {
        "id": 22,
        "title": "Min Stack",
        "category": "Stack",
        "difficulty": "Medium",
        "description": """\
Design a stack that supports `push`, `pop`, `top`, and retrieving the minimum element in constant time.

Implement `MinStack` with methods: `push(val)`, `pop()`, `top()`, `get_min()`.
""",
        "python_tips": "Hint: maintain a second stack that tracks the current minimum at each level.",
        "starter_code": "class MinStack:\n    def __init__(self): pass\n    def push(self, val): pass\n    def pop(self): pass\n    def top(self): pass\n    def get_min(self): pass\n",
        "test_cases": [
            {
                "class_test": True,
                "class_name": "MinStack",
                "operations": ["MinStack", "push", "push", "push", "get_min", "pop", "top", "get_min"],
                "arguments":  [[], [-2], [0], [-3], [], [], [], []],
                "expected":   [None, None, None, None, -3, None, 0, -2],
            },
        ],
        "solution": """\
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    def push(self, val):
        self.stack.append(val)
        min_val = min(val, self.min_stack[-1] if self.min_stack else val)
        self.min_stack.append(min_val)
    def pop(self):
        self.stack.pop()
        self.min_stack.pop()
    def top(self):
        return self.stack[-1]
    def get_min(self):
        return self.min_stack[-1]
""",
    },
    {
        "id": 23,
        "title": "Evaluate Reverse Polish Notation",
        "category": "Stack",
        "difficulty": "Medium",
        "description": """\
Evaluate the value of an arithmetic expression in Reverse Polish Notation (postfix).

**Example:**
```
Input:  tokens = ["2","1","+","3","*"]
Output: 9    # ((2 + 1) * 3) = 9
```
""",
        "python_tips": "Hint: push numbers onto a stack; on an operator, pop two values, compute, push result.",
        "starter_code": "def eval_rpn(tokens):\n    pass\n",
        "test_cases": [
            {"input": (["2", "1", "+", "3", "*"],), "expected": 9},
            {"input": (["4", "13", "5", "/", "+"],), "expected": 6},
        ],
        "solution": """\
def eval_rpn(tokens):
    stack = []
    ops = {'+': lambda a,b: a+b, '-': lambda a,b: a-b,
           '*': lambda a,b: a*b, '/': lambda a,b: int(a/b)}
    for t in tokens:
        if t in ops:
            b, a = stack.pop(), stack.pop()
            stack.append(ops[t](a, b))
        else:
            stack.append(int(t))
    return stack[0]
""",
    },
    {
        "id": 24,
        "title": "Generate Parentheses",
        "category": "Stack",
        "difficulty": "Medium",
        "description": """\
Given `n` pairs of parentheses, generate all combinations of well-formed parentheses.

**Example:**
```
Input:  n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
```
""",
        "python_tips": "Hint: backtracking with a stack — track open and close counts; add `(` if open < n, `)` if close < open.",
        "starter_code": "def generate_parenthesis(n):\n    pass\n",
        "test_cases": [
            {"input": (1,), "expected": ["()"]},
            {"input": (3,), "expected": ["((()))","(()())","(())()","()(())","()()()"], "unordered": True},
        ],
        "solution": """\
def generate_parenthesis(n):
    result = []
    def bt(s, open, close):
        if len(s) == 2 * n:
            result.append(s)
            return
        if open < n:
            bt(s + '(', open + 1, close)
        if close < open:
            bt(s + ')', open, close + 1)
    bt('', 0, 0)
    return result
""",
    },
    {
        "id": 25,
        "title": "Daily Temperatures",
        "category": "Stack",
        "difficulty": "Medium",
        "description": """\
Given an array `temperatures`, return an array `answer` where `answer[i]` is the number of days you have to wait after day `i` to get a warmer temperature. If there's no future day with warmer temperature, set `answer[i] = 0`.

**Example:**
```
Input:  temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]
```
""",
        "python_tips": "Hint: use a monotonic stack storing indices of temperatures waiting for a warmer day.",
        "starter_code": "def daily_temperatures(temperatures):\n    pass\n",
        "test_cases": [
            {"input": ([73,74,75,71,69,72,76,73],), "expected": [1,1,4,2,1,1,0,0]},
            {"input": ([30,40,50,60],), "expected": [1,1,1,0]},
        ],
        "solution": """\
def daily_temperatures(temperatures):
    result = [0] * len(temperatures)
    stack = []
    for i, t in enumerate(temperatures):
        while stack and t > temperatures[stack[-1]]:
            idx = stack.pop()
            result[idx] = i - idx
        stack.append(i)
    return result
""",
    },
    {
        "id": 26,
        "title": "Car Fleet",
        "category": "Stack",
        "difficulty": "Medium",
        "description": """\
N cars are heading to the same destination. Given `position` and `speed` arrays, a car fleet is a group of cars that arrive at the destination together. Return the number of car fleets.

**Example:**
```
Input:  target=12, position=[10,8,0,5,3], speed=[2,4,1,1,3]
Output: 3
```
""",
        "python_tips": "Hint: sort by position descending, compute time to reach target. If next car is slower, it joins the fleet.",
        "starter_code": "def car_fleet(target, position, speed):\n    pass\n",
        "test_cases": [
            {"input": (12, [10,8,0,5,3], [2,4,1,1,3]), "expected": 3},
            {"input": (10, [3], [3]), "expected": 1},
        ],
        "solution": """\
def car_fleet(target, position, speed):
    pairs = sorted(zip(position, speed), reverse=True)
    stack = []
    for pos, spd in pairs:
        time = (target - pos) / spd
        if not stack or time > stack[-1]:
            stack.append(time)
    return len(stack)
""",
    },
    {
        "id": 27,
        "title": "Largest Rectangle In Histogram",
        "category": "Stack",
        "difficulty": "Hard",
        "description": """\
Given an array of integers `heights` representing the histogram's bar heights, return the area of the largest rectangle in the histogram.

**Example:**
```
Input:  heights = [2,1,5,6,2,3]
Output: 10
```
""",
        "python_tips": "Hint: monotonic increasing stack. When a bar is shorter than the top of the stack, pop and calculate the area.",
        "starter_code": "def largest_rectangle_area(heights):\n    pass\n",
        "test_cases": [
            {"input": ([2,1,5,6,2,3],), "expected": 10},
            {"input": ([2,4],), "expected": 4},
        ],
        "solution": """\
def largest_rectangle_area(heights):
    stack, max_area = [], 0
    for i, h in enumerate(heights + [0]):
        start = i
        while stack and stack[-1][1] > h:
            idx, height = stack.pop()
            max_area = max(max_area, height * (i - idx))
            start = idx
        stack.append((start, h))
    return max_area
""",
    },

    # ── Binary Search (stubs) ─────────────────────────────────────
    {
        "id": 28,
        "title": "Search a 2D Matrix",
        "category": "Binary Search",
        "difficulty": "Medium",
        "description": """\
Write an efficient algorithm to search for a value `target` in an m × n matrix. Each row is sorted; the first integer of each row is greater than the last of the previous row.

**Example:**
```
Input:  matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: True
```
""",
        "python_tips": "Hint: treat the matrix as a flat sorted array. Row = mid // cols, Col = mid % cols.",
        "starter_code": "def search_matrix(matrix, target):\n    pass\n",
        "test_cases": [
            {"input": ([[1,3,5,7],[10,11,16,20],[23,30,34,60]], 3), "expected": True},
            {"input": ([[1,3,5,7],[10,11,16,20],[23,30,34,60]], 13), "expected": False},
        ],
        "solution": """\
def search_matrix(matrix, target):
    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1
    while left <= right:
        mid = (left + right) // 2
        val = matrix[mid // cols][mid % cols]
        if val == target: return True
        elif val < target: left = mid + 1
        else: right = mid - 1
    return False
""",
    },
    {
        "id": 29,
        "title": "Koko Eating Bananas",
        "category": "Binary Search",
        "difficulty": "Medium",
        "description": """\
Koko can eat at most `k` bananas per hour. Given piles of bananas and `h` hours, find the minimum `k` such that she can eat all bananas within `h` hours.

**Example:**
```
Input:  piles = [3,6,7,11], h = 8
Output: 4
```
""",
        "python_tips": "Hint: binary search on the answer (k). For a given k, hours_needed = sum(ceil(pile/k) for pile in piles).",
        "starter_code": "def min_eating_speed(piles, h):\n    pass\n",
        "test_cases": [
            {"input": ([3,6,7,11], 8), "expected": 4},
            {"input": ([30,11,23,4,20], 5), "expected": 30},
        ],
        "solution": """\
def min_eating_speed(piles, h):
    import math
    left, right = 1, max(piles)
    while left < right:
        mid = (left + right) // 2
        if sum(math.ceil(p / mid) for p in piles) <= h:
            right = mid
        else:
            left = mid + 1
    return left
""",
    },
    {
        "id": 30,
        "title": "Find Minimum In Rotated Sorted Array",
        "category": "Binary Search",
        "difficulty": "Medium",
        "description": "Given a rotated sorted array, find the minimum element in O(log n) time.",
        "python_tips": "Hint: binary search comparing mid to right. If nums[mid] > nums[right], min is in the right half.",
        "starter_code": "def find_min(nums):\n    pass\n",
        "test_cases": [
            {"input": ([3,4,5,1,2],), "expected": 1},
            {"input": ([4,5,6,7,0,1,2],), "expected": 0},
            {"input": ([11,13,15,17],), "expected": 11},
        ],
        "solution": """\
def find_min(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    return nums[left]
""",
    },
    {
        "id": 31,
        "title": "Search In Rotated Sorted Array",
        "category": "Binary Search",
        "difficulty": "Medium",
        "description": "Given a rotated sorted array with no duplicates, search for `target` in O(log n). Return index or -1.",
        "python_tips": "Hint: standard binary search, but determine which half is sorted before deciding which direction to go.",
        "starter_code": "def search(nums, target):\n    pass\n",
        "test_cases": [
            {"input": ([4,5,6,7,0,1,2], 0), "expected": 4},
            {"input": ([4,5,6,7,0,1,2], 3), "expected": -1},
        ],
        "solution": """\
def search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target: return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
""",
    },
    {
        "id": 32,
        "title": "Time Based Key-Value Store",
        "category": "Binary Search",
        "difficulty": "Medium",
        "description": "Design a time-based key-value store supporting set(key, value, timestamp) and get(key, timestamp) returning the most recent value at or before timestamp.",
        "python_tips": "Hint: store list of (timestamp, value) per key, then binary search for the closest timestamp.",
        "starter_code": "class TimeMap:\n    def __init__(self): pass\n    def set(self, key, value, timestamp): pass\n    def get(self, key, timestamp): pass\n",
        "test_cases": [
            {
                "class_test": True,
                "class_name": "TimeMap",
                "operations": ["TimeMap", "set", "get", "get", "set", "get", "get"],
                "arguments": [[], ["foo", "bar", 1], ["foo", 1], ["foo", 3], ["foo", "bar2", 4], ["foo", 4], ["foo", 5]],
                "expected": [None, None, "bar", "bar", None, "bar2", "bar2"],
            },
        ],
        "solution": """\
class TimeMap:
    def __init__(self):
        self.store = {}
    def set(self, key, value, timestamp):
        self.store.setdefault(key, []).append((timestamp, value))
    def get(self, key, timestamp):
        import bisect
        if key not in self.store: return ""
        entries = self.store[key]
        idx = bisect.bisect_right(entries, (timestamp, chr(127))) - 1
        return entries[idx][1] if idx >= 0 else ""
""",
    },
    {
        "id": 33,
        "title": "Median of Two Sorted Arrays",
        "category": "Binary Search",
        "difficulty": "Hard",
        "description": "Given two sorted arrays, find the median of the combined sorted array in O(log(m+n)) time.",
        "python_tips": "Hint: binary search on the smaller array to find the correct partition point.",
        "starter_code": "def find_median_sorted_arrays(nums1, nums2):\n    pass\n",
        "test_cases": [
            {"input": ([1,3], [2]), "expected": 2.0},
            {"input": ([1,2], [3,4]), "expected": 2.5},
        ],
        "solution": """\
def find_median_sorted_arrays(nums1, nums2):
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    m, n = len(nums1), len(nums2)
    half = (m + n) // 2
    left, right = 0, m
    while True:
        i = (left + right) // 2
        j = half - i
        lmax1 = nums1[i-1] if i > 0 else float('-inf')
        rmin1 = nums1[i]   if i < m else float('inf')
        lmax2 = nums2[j-1] if j > 0 else float('-inf')
        rmin2 = nums2[j]   if j < n else float('inf')
        if lmax1 <= rmin2 and lmax2 <= rmin1:
            if (m + n) % 2:
                return float(min(rmin1, rmin2))
            return (max(lmax1, lmax2) + min(rmin1, rmin2)) / 2
        elif lmax1 > rmin2:
            right = i - 1
        else:
            left = i + 1
""",
    },

    # ── Linked List ───────────────────────────────────────────────
    {
        "id": 34,
        "title": "Reverse Linked List",
        "category": "Linked List",
        "difficulty": "Easy",
        "description": "Given the head of a singly linked list, reverse it and return the new head.",
        "python_tips": "Hint: use three pointers: prev=None, curr=head, next_node. Re-link curr.next = prev, advance all three.",
        "starter_code": "class ListNode:\n    def __init__(self, val=0, next=None):\n        self.val=val; self.next=next\n\ndef reverse_list(head):\n    pass\n",
        "test_cases": [
            {"input": ("list:[1,2,3,4,5]",), "expected": "list:[5,4,3,2,1]", "is_list": True},
            {"input": ("list:[1,2]",), "expected": "list:[2,1]", "is_list": True},
            {"input": ("list:[]",), "expected": "list:[]", "is_list": True},
        ],
        "solution": """\
def reverse_list(head):
    prev, curr = None, head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev
""",
    },
    {
        "id": 35,
        "title": "Merge Two Sorted Lists",
        "category": "Linked List",
        "difficulty": "Easy",
        "description": "Merge two sorted linked lists into one sorted list.",
        "python_tips": "Hint: use a dummy head node. Compare current nodes of both lists; link the smaller one.",
        "starter_code": "def merge_two_lists(list1, list2):\n    pass\n",
        "test_cases": [
            {"input": ("list:[1,2,4]", "list:[1,3,4]"), "expected": "list:[1,1,2,3,4,4]", "is_list": True},
            {"input": ("list:[]", "list:[]"), "expected": "list:[]", "is_list": True},
            {"input": ("list:[]", "list:[0]"), "expected": "list:[0]", "is_list": True},
        ],
        "solution": """\
def merge_two_lists(list1, list2):
    dummy = curr = type('N', (), {'next': None})()
    while list1 and list2:
        if list1.val <= list2.val:
            curr.next, list1 = list1, list1.next
        else:
            curr.next, list2 = list2, list2.next
        curr = curr.next
    curr.next = list1 or list2
    return dummy.next
""",
    },
    {
        "id": 36,
        "title": "Reorder List",
        "category": "Linked List",
        "difficulty": "Medium",
        "description": "Given L0→L1→…→Ln, reorder it to L0→Ln→L1→Ln-1→…",
        "python_tips": "Hint: find midpoint (slow/fast pointer), reverse the second half, then merge both halves.",
        "starter_code": "def reorder_list(head):\n    pass\n",
        "test_cases": [
            {"input": ("list:[1,2,3,4]",), "expected": "list:[1,4,2,3]", "is_list": True, "check_head": True},
            {"input": ("list:[1,2,3,4,5]",), "expected": "list:[1,5,2,4,3]", "is_list": True, "check_head": True},
        ],
        "solution": """\
def reorder_list(head):
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    second = slow.next
    slow.next = None
    prev = None
    while second:
        tmp = second.next
        second.next = prev
        prev = second
        second = tmp
    first, second = head, prev
    while second:
        tmp1, tmp2 = first.next, second.next
        first.next = second
        second.next = tmp1
        first = tmp1
        second = tmp2
""",
    },
    {
        "id": 37,
        "title": "Remove Nth Node From End of List",
        "category": "Linked List",
        "difficulty": "Medium",
        "description": "Remove the nth node from the end of the linked list in one pass.",
        "python_tips": "Hint: two pointers separated by n steps. When fast reaches the end, slow is at the node to remove.",
        "starter_code": "def remove_nth_from_end(head, n):\n    pass\n",
        "test_cases": [
            {"input": ("list:[1,2,3,4,5]", 2), "expected": "list:[1,2,3,5]", "is_list": True},
            {"input": ("list:[1]", 1), "expected": "list:[]", "is_list": True},
            {"input": ("list:[1,2]", 1), "expected": "list:[1]", "is_list": True},
        ],
        "solution": """\
def remove_nth_from_end(head, n):
    dummy = type('N', (), {'next': head, 'val': 0})()
    fast = slow = dummy
    for _ in range(n + 1):
        fast = fast.next
    while fast:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return dummy.next
""",
    },
    {
        "id": 38,
        "title": "Copy List With Random Pointer",
        "category": "Linked List",
        "difficulty": "Medium",
        "description": "Deep copy a linked list where each node also has a random pointer.",
        "python_tips": "Hint: use a hash map from original node to its copy. First pass creates copies, second pass links them.",
        "starter_code": "def copy_random_list(head):\n    pass\n",
        "test_cases": [],
        "solution": """\
def copy_random_list(head):
    if not head: return None
    old_to_new = {}
    curr = head
    while curr:
        old_to_new[curr] = type(curr)(curr.val)
        curr = curr.next
    curr = head
    while curr:
        if curr.next:   old_to_new[curr].next   = old_to_new[curr.next]
        if curr.random: old_to_new[curr].random = old_to_new[curr.random]
        curr = curr.next
    return old_to_new[head]
""",
    },
    {
        "id": 39,
        "title": "Add Two Numbers",
        "category": "Linked List",
        "difficulty": "Medium",
        "description": "Two non-empty linked lists represent non-negative integers in reverse order. Add them and return the sum as a linked list.",
        "python_tips": "Hint: simulate grade-school addition with a `carry` variable. Advance both pointers simultaneously.",
        "starter_code": "def add_two_numbers(l1, l2):\n    pass\n",
        "test_cases": [
            {"input": ("list:[2,4,3]", "list:[5,6,4]"), "expected": "list:[7,0,8]", "is_list": True},
            {"input": ("list:[0]", "list:[0]"), "expected": "list:[0]", "is_list": True},
            {"input": ("list:[9,9,9,9,9,9,9]", "list:[9,9,9,9]"), "expected": "list:[8,9,9,9,0,0,0,1]", "is_list": True},
        ],
        "solution": """\
def add_two_numbers(l1, l2):
    dummy = curr = type('N', (), {'val':0,'next':None})()
    carry = 0
    while l1 or l2 or carry:
        v1 = l1.val if l1 else 0
        v2 = l2.val if l2 else 0
        total = v1 + v2 + carry
        carry = total // 10
        curr.next = type('N', (), {'val': total % 10, 'next': None})()
        curr = curr.next
        if l1: l1 = l1.next
        if l2: l2 = l2.next
    return dummy.next
""",
    },
    {
        "id": 40,
        "title": "Linked List Cycle",
        "category": "Linked List",
        "difficulty": "Easy",
        "description": "Given the head of a linked list, determine if there is a cycle.",
        "python_tips": "Hint: Floyd's tortoise & hare — slow moves 1 step, fast moves 2. If they meet, there's a cycle.",
        "starter_code": "def has_cycle(head):\n    pass\n",
        "test_cases": [
            {"input": ("list:[3,2,0,-4]",), "expected": True,  "is_list": True, "cycle_pos": 1},
            {"input": ("list:[1,2]",),       "expected": True,  "is_list": True, "cycle_pos": 0},
            {"input": ("list:[1]",),          "expected": False, "is_list": True, "cycle_pos": -1},
        ],
        "solution": """\
def has_cycle(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False
""",
    },
    {
        "id": 41,
        "title": "Find The Duplicate Number",
        "category": "Linked List",
        "difficulty": "Medium",
        "description": "Given an array of n+1 integers where each integer is between 1 and n, find the duplicate.",
        "python_tips": "Hint: treat array values as linked list pointers. Use Floyd's cycle detection.",
        "starter_code": "def find_duplicate(nums):\n    pass\n",
        "test_cases": [
            {"input": ([1,3,4,2,2],), "expected": 2},
            {"input": ([3,1,3,4,2],), "expected": 3},
        ],
        "solution": """\
def find_duplicate(nums):
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast: break
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    return slow
""",
    },
    {
        "id": 42,
        "title": "LRU Cache",
        "category": "Linked List",
        "difficulty": "Medium",
        "description": "Design a data structure that follows the LRU (Least Recently Used) cache eviction policy with O(1) get and put.",
        "python_tips": "Hint: use an OrderedDict or a doubly-linked list + hash map.",
        "starter_code": "class LRUCache:\n    def __init__(self, capacity): pass\n    def get(self, key): pass\n    def put(self, key, value): pass\n",
        "test_cases": [
            {
                "class_test": True,
                "class_name": "LRUCache",
                "operations": ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"],
                "arguments": [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]],
                "expected": [None, None, None, 1, None, -1, None, -1, 3, 4],
            },
        ],
        "solution": """\
class LRUCache:
    def __init__(self, capacity):
        from collections import OrderedDict
        self.cap = capacity
        self.cache = OrderedDict()
    def get(self, key):
        if key not in self.cache: return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    def put(self, key, value):
        if key in self.cache: self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
""",
    },
    {
        "id": 43,
        "title": "Merge K Sorted Lists",
        "category": "Linked List",
        "difficulty": "Hard",
        "description": "Merge k sorted linked lists into one sorted list.",
        "python_tips": "Hint: use a min-heap of (value, index, node) tuples for efficient minimum retrieval.",
        "starter_code": "def merge_k_lists(lists):\n    pass\n",
        "test_cases": [
            {"input": (["list:[1,4,5]", "list:[1,3,4]", "list:[2,6]"],), "expected": "list:[1,1,2,3,4,4,5,6]", "is_list": True},
            {"input": ([],), "expected": "list:[]", "is_list": True},
        ],
        "solution": """\
def merge_k_lists(lists):
    import heapq
    dummy = curr = type('N', (), {'val':0,'next':None})()
    heap = []
    for i, node in enumerate(lists):
        if node: heapq.heappush(heap, (node.val, i, node))
    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next: heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next
""",
    },
    {
        "id": 44,
        "title": "Reverse Nodes In K Group",
        "category": "Linked List",
        "difficulty": "Hard",
        "description": "Given a linked list, reverse nodes in groups of k and return the modified list.",
        "python_tips": "Hint: check if k nodes remain, reverse them, then recursively handle the rest.",
        "starter_code": "def reverse_k_group(head, k):\n    pass\n",
        "test_cases": [
            {"input": ("list:[1,2,3,4,5]", 2), "expected": "list:[2,1,4,3,5]", "is_list": True},
            {"input": ("list:[1,2,3,4,5]", 3), "expected": "list:[3,2,1,4,5]", "is_list": True},
        ],
        "solution": """\
def reverse_k_group(head, k):
    node, count = head, 0
    while node and count < k:
        node = node.next
        count += 1
    if count < k: return head
    prev, curr = None, head
    for _ in range(k):
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    head.next = reverse_k_group(curr, k)
    return prev
""",
    },

    # ── Trees ─────────────────────────────────────────────────────
    {
        "id": 45,
        "title": "Maximum Depth of Binary Tree",
        "category": "Trees",
        "difficulty": "Easy",
        "description": "Given the root of a binary tree, return its maximum depth.",
        "python_tips": "Hint: recursion — max depth = 1 + max(depth(left), depth(right)). Base case: None → 0.",
        "starter_code": "def max_depth(root):\n    pass\n",
        "test_cases": [
            {"input": ("tree:[3,9,20,null,null,15,7]",), "expected": 3, "is_tree": True},
            {"input": ("tree:[1,null,2]",), "expected": 2, "is_tree": True},
            {"input": ("tree:[]",), "expected": 0, "is_tree": True},
        ],
        "solution": """\
def max_depth(root):
    if not root: return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
""",
    },
    {
        "id": 46,
        "title": "Diameter of Binary Tree",
        "category": "Trees",
        "difficulty": "Easy",
        "description": "Given the root of a binary tree, return the length of the diameter (longest path between any two nodes).",
        "python_tips": "Hint: for each node, diameter through it = depth(left) + depth(right). Track the global max.",
        "starter_code": "def diameter_of_binary_tree(root):\n    pass\n",
        "test_cases": [
            {"input": ("tree:[1,2,3,4,5]",), "expected": 3, "is_tree": True},
            {"input": ("tree:[1,2]",), "expected": 1, "is_tree": True},
        ],
        "solution": """\
def diameter_of_binary_tree(root):
    res = [0]
    def depth(node):
        if not node: return 0
        l, r = depth(node.left), depth(node.right)
        res[0] = max(res[0], l + r)
        return 1 + max(l, r)
    depth(root)
    return res[0]
""",
    },
    {
        "id": 47,
        "title": "Balanced Binary Tree",
        "category": "Trees",
        "difficulty": "Easy",
        "description": "Determine if a binary tree is height-balanced (depth of subtrees never differ by more than 1).",
        "python_tips": "Hint: recursive DFS returning height; return -1 to signal imbalance early.",
        "starter_code": "def is_balanced(root):\n    pass\n",
        "test_cases": [
            {"input": ("tree:[3,9,20,null,null,15,7]",), "expected": True, "is_tree": True},
            {"input": ("tree:[1,2,2,3,3,null,null,4,4]",), "expected": False, "is_tree": True},
            {"input": ("tree:[]",), "expected": True, "is_tree": True},
        ],
        "solution": """\
def is_balanced(root):
    def height(node):
        if not node: return 0
        l, r = height(node.left), height(node.right)
        if l == -1 or r == -1 or abs(l - r) > 1: return -1
        return 1 + max(l, r)
    return height(root) != -1
""",
    },
    {
        "id": 48,
        "title": "Same Tree",
        "category": "Trees",
        "difficulty": "Easy",
        "description": "Given the roots of two binary trees, check if they are the same.",
        "python_tips": "Hint: recursively compare values and both subtrees.",
        "starter_code": "def is_same_tree(p, q):\n    pass\n",
        "test_cases": [
            {"input": ("tree:[1,2,3]", "tree:[1,2,3]"), "expected": True, "is_tree": True},
            {"input": ("tree:[1,2]", "tree:[1,null,2]"), "expected": False, "is_tree": True},
            {"input": ("tree:[]", "tree:[]"), "expected": True, "is_tree": True},
        ],
        "solution": """\
def is_same_tree(p, q):
    if not p and not q: return True
    if not p or not q or p.val != q.val: return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)
""",
    },
    {
        "id": 49,
        "title": "Subtree of Another Tree",
        "category": "Trees",
        "difficulty": "Easy",
        "description": "Given the roots of two binary trees `root` and `subRoot`, return True if `subRoot` is a subtree of `root`.",
        "python_tips": "Hint: for each node in root, check if the subtree rooted there equals subRoot using isSameTree.",
        "starter_code": "def is_subtree(root, sub_root):\n    pass\n",
        "test_cases": [
            {"input": ("tree:[3,4,5,1,2]", "tree:[4,1,2]"), "expected": True, "is_tree": True},
            {"input": ("tree:[3,4,5,1,2,null,null,null,null,0]", "tree:[4,1,2]"), "expected": False, "is_tree": True},
        ],
        "solution": """\
def is_subtree(root, sub_root):
    def same(p, q):
        if not p and not q: return True
        if not p or not q or p.val != q.val: return False
        return same(p.left, q.left) and same(p.right, q.right)
    if not root: return False
    if same(root, sub_root): return True
    return is_subtree(root.left, sub_root) or is_subtree(root.right, sub_root)
""",
    },
    {
        "id": 50,
        "title": "Lowest Common Ancestor of a BST",
        "category": "Trees",
        "difficulty": "Medium",
        "description": "Given a BST and two nodes p and q, find their lowest common ancestor.",
        "python_tips": "Hint: in a BST, if both p and q are less than node, go left. If both greater, go right. Otherwise, current node is the LCA.",
        "starter_code": "def lowest_common_ancestor(root, p, q):\n    pass\n",
        "test_cases": [
            {"input": ("tree:[6,2,8,0,4,null,null,null,null,3,5]", "treenode:2", "treenode:8"), "expected": 6, "is_tree": True},
            {"input": ("tree:[6,2,8,0,4,null,null,null,null,3,5]", "treenode:2", "treenode:4"), "expected": 2, "is_tree": True},
            {"input": ("tree:[2,1]", "treenode:2", "treenode:1"), "expected": 2, "is_tree": True},
        ],
        "solution": """\
def lowest_common_ancestor(root, p, q):
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root
""",
    },
    {
        "id": 51,
        "title": "Binary Tree Level Order Traversal",
        "category": "Trees",
        "difficulty": "Medium",
        "description": "Return the level-order traversal of a binary tree's nodes' values (i.e., from left to right, level by level).",
        "python_tips": "Hint: BFS using a queue (collections.deque). Process all nodes at the current level before moving to the next.",
        "starter_code": "def level_order(root):\n    pass\n",
        "test_cases": [
            {"input": ("tree:[3,9,20,null,null,15,7]",), "expected": [[3],[9,20],[15,7]], "is_tree": True},
            {"input": ("tree:[1]",), "expected": [[1]], "is_tree": True},
            {"input": ("tree:[]",), "expected": [], "is_tree": True},
        ],
        "solution": """\
def level_order(root):
    from collections import deque
    if not root: return []
    result, queue = [], deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:  queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result
""",
    },
    {
        "id": 52,
        "title": "Binary Tree Right Side View",
        "category": "Trees",
        "difficulty": "Medium",
        "description": "Imagine standing on the right side of a binary tree. Return the values you can see.",
        "python_tips": "Hint: BFS level order traversal — take the last node of each level.",
        "starter_code": "def right_side_view(root):\n    pass\n",
        "test_cases": [
            {"input": ("tree:[1,2,3,null,5,null,4]",), "expected": [1,3,4], "is_tree": True},
            {"input": ("tree:[1,null,3]",), "expected": [1,3], "is_tree": True},
            {"input": ("tree:[]",), "expected": [], "is_tree": True},
        ],
        "solution": """\
def right_side_view(root):
    from collections import deque
    if not root: return []
    result, q = [], deque([root])
    while q:
        for i in range(len(q)):
            node = q.popleft()
            if i == len(q): result.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
        if not result or result[-1] != node.val:
            result.append(node.val)
    return result
""",
    },
    {
        "id": 53,
        "title": "Count Good Nodes In Binary Tree",
        "category": "Trees",
        "difficulty": "Medium",
        "description": "A node X is 'good' if there are no nodes with a value greater than X on the path from root to X. Count good nodes.",
        "python_tips": "Hint: DFS passing the max value seen so far along the path.",
        "starter_code": "def good_nodes(root):\n    pass\n",
        "test_cases": [
            {"input": ("tree:[3,1,4,3,null,1,5]",), "expected": 4, "is_tree": True},
            {"input": ("tree:[3,3,null,4,2]",), "expected": 3, "is_tree": True},
            {"input": ("tree:[1]",), "expected": 1, "is_tree": True},
        ],
        "solution": """\
def good_nodes(root):
    def dfs(node, max_val):
        if not node: return 0
        good = 1 if node.val >= max_val else 0
        max_val = max(max_val, node.val)
        return good + dfs(node.left, max_val) + dfs(node.right, max_val)
    return dfs(root, root.val)
""",
    },
    {
        "id": 54,
        "title": "Validate Binary Search Tree",
        "category": "Trees",
        "difficulty": "Medium",
        "description": "Determine if a binary tree is a valid BST.",
        "python_tips": "Hint: DFS with min/max bounds. Each node must satisfy min < node.val < max.",
        "starter_code": "def is_valid_bst(root):\n    pass\n",
        "test_cases": [
            {"input": ("tree:[2,1,3]",), "expected": True, "is_tree": True},
            {"input": ("tree:[5,1,4,null,null,3,6]",), "expected": False, "is_tree": True},
            {"input": ("tree:[1]",), "expected": True, "is_tree": True},
        ],
        "solution": """\
def is_valid_bst(root):
    def validate(node, lo, hi):
        if not node: return True
        if not (lo < node.val < hi): return False
        return validate(node.left, lo, node.val) and validate(node.right, node.val, hi)
    return validate(root, float('-inf'), float('inf'))
""",
    },
    {
        "id": 55,
        "title": "Kth Smallest Element in a BST",
        "category": "Trees",
        "difficulty": "Medium",
        "description": "Given the root of a BST and an integer k, return the kth smallest value.",
        "python_tips": "Hint: in-order traversal of a BST gives sorted order. Count nodes as you visit.",
        "starter_code": "def kth_smallest(root, k):\n    pass\n",
        "test_cases": [
            {"input": ("tree:[3,1,4,null,2]", 1), "expected": 1, "is_tree": True},
            {"input": ("tree:[5,3,6,2,4,null,null,1]", 3), "expected": 3, "is_tree": True},
        ],
        "solution": """\
def kth_smallest(root, k):
    stack, curr = [], root
    while stack or curr:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        k -= 1
        if k == 0: return curr.val
        curr = curr.right
""",
    },
    {
        "id": 56,
        "title": "Construct Binary Tree from Preorder and Inorder Traversal",
        "category": "Trees",
        "difficulty": "Medium",
        "description": "Given preorder and inorder traversal arrays, reconstruct the binary tree.",
        "python_tips": "Hint: preorder[0] is root. Find it in inorder to split left/right subtrees. Recurse.",
        "starter_code": "def build_tree(preorder, inorder):\n    pass\n",
        "test_cases": [
            {"input": ([3,9,20,15,7], [9,3,15,20,7]), "expected": "tree:[3,9,20,null,null,15,7]", "is_tree": True},
            {"input": ([-1], [-1]), "expected": "tree:[-1]", "is_tree": True},
        ],
        "solution": """\
def build_tree(preorder, inorder):
    if not preorder: return None
    root_val = preorder[0]
    mid = inorder.index(root_val)
    root = type('TreeNode', (), {'val': root_val, 'left': None, 'right': None})()
    root.left  = build_tree(preorder[1:mid+1], inorder[:mid])
    root.right = build_tree(preorder[mid+1:],  inorder[mid+1:])
    return root
""",
    },
    {
        "id": 57,
        "title": "Binary Tree Maximum Path Sum",
        "category": "Trees",
        "difficulty": "Hard",
        "description": "Find the maximum path sum in a binary tree (path can start and end at any node).",
        "python_tips": "Hint: DFS returning max gain from each subtree. Track global max including both children.",
        "starter_code": "def max_path_sum(root):\n    pass\n",
        "test_cases": [
            {"input": ("tree:[1,2,3]",), "expected": 6, "is_tree": True},
            {"input": ("tree:[-10,9,20,null,null,15,7]",), "expected": 42, "is_tree": True},
            {"input": ("tree:[-3]",), "expected": -3, "is_tree": True},
        ],
        "solution": """\
def max_path_sum(root):
    res = [root.val]
    def dfs(node):
        if not node: return 0
        left  = max(dfs(node.left),  0)
        right = max(dfs(node.right), 0)
        res[0] = max(res[0], node.val + left + right)
        return node.val + max(left, right)
    dfs(root)
    return res[0]
""",
    },
    {
        "id": 58,
        "title": "Serialize and Deserialize Binary Tree",
        "category": "Trees",
        "difficulty": "Hard",
        "description": "Design an algorithm to serialize and deserialize a binary tree.",
        "python_tips": "Hint: use BFS or DFS with null markers. Store values separated by commas.",
        "starter_code": "def serialize(root):\n    pass\n\ndef deserialize(data):\n    pass\n",
        "test_cases": [
            {"input": ("tree:[1,2,3,null,null,4,5]",), "expected": "tree:[1,2,3,null,null,4,5]", "is_tree": True, "encode_decode": True},
            {"input": ("tree:[]",), "expected": "tree:[]", "is_tree": True, "encode_decode": True},
        ],
        "solution": """\
def serialize(root):
    from collections import deque
    if not root: return ''
    result, q = [], deque([root])
    while q:
        node = q.popleft()
        if node:
            result.append(str(node.val))
            q.append(node.left)
            q.append(node.right)
        else:
            result.append('N')
    return ','.join(result)

def deserialize(data):
    from collections import deque
    if not data: return None
    vals = data.split(',')
    root = type('T', (), {'val': int(vals[0]), 'left': None, 'right': None})()
    q = deque([root])
    i = 1
    while q:
        node = q.popleft()
        if vals[i] != 'N':
            node.left = type('T', (), {'val': int(vals[i]), 'left': None, 'right': None})()
            q.append(node.left)
        i += 1
        if vals[i] != 'N':
            node.right = type('T', (), {'val': int(vals[i]), 'left': None, 'right': None})()
            q.append(node.right)
        i += 1
    return root
""",
    },

    # ── Tries ─────────────────────────────────────────────────────
    {
        "id": 59,
        "title": "Implement Trie (Prefix Tree)",
        "category": "Tries",
        "difficulty": "Medium",
        "description": "Implement a trie with insert, search, and startsWith methods.",
        "python_tips": "Hint: each node is a dict of children + a boolean `is_end`. Walk character by character.",
        "starter_code": "class Trie:\n    def __init__(self): pass\n    def insert(self, word): pass\n    def search(self, word): pass\n    def starts_with(self, prefix): pass\n",
        "test_cases": [
            {
                "class_test": True,
                "class_name": "Trie",
                "operations": ["Trie", "insert", "search", "search", "starts_with", "insert", "search"],
                "arguments": [[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]],
                "expected": [None, None, True, False, True, None, True],
            },
        ],
        "solution": """\
class Trie:
    def __init__(self): self.root = {}
    def insert(self, word):
        node = self.root
        for c in word:
            node = node.setdefault(c, {})
        node['#'] = True
    def search(self, word):
        node = self.root
        for c in word:
            if c not in node: return False
            node = node[c]
        return '#' in node
    def starts_with(self, prefix):
        node = self.root
        for c in prefix:
            if c not in node: return False
            node = node[c]
        return True
""",
    },
    {
        "id": 60,
        "title": "Design Add and Search Words Data Structure",
        "category": "Tries",
        "difficulty": "Medium",
        "description": "Design a data structure that supports addWord(word) and search(word) where '.' can match any letter.",
        "python_tips": "Hint: trie + DFS for '.' wildcard — try all children when you encounter a dot.",
        "starter_code": "class WordDictionary:\n    def __init__(self): pass\n    def add_word(self, word): pass\n    def search(self, word): pass\n",
        "test_cases": [
            {
                "class_test": True,
                "class_name": "WordDictionary",
                "operations": ["WordDictionary", "add_word", "add_word", "add_word", "search", "search", "search", "search"],
                "arguments": [[], ["bad"], ["dad"], ["mad"], ["pad"], ["bad"], [".ad"], ["b.."]],
                "expected": [None, None, None, None, False, True, True, True],
            },
        ],
        "solution": """\
class WordDictionary:
    def __init__(self): self.root = {}
    def add_word(self, word):
        node = self.root
        for c in word: node = node.setdefault(c, {})
        node['#'] = True
    def search(self, word):
        def dfs(node, i):
            if i == len(word): return '#' in node
            c = word[i]
            if c == '.':
                return any(dfs(child, i+1) for k, child in node.items() if k != '#')
            if c not in node: return False
            return dfs(node[c], i+1)
        return dfs(self.root, 0)
""",
    },
    {
        "id": 61,
        "title": "Word Search II",
        "category": "Tries",
        "difficulty": "Hard",
        "description": "Given a board of characters and a list of words, return all words that can be found in the board.",
        "python_tips": "Hint: build a trie of all words, then DFS from each cell on the board matching trie paths.",
        "starter_code": "def find_words(board, words):\n    pass\n",
        "test_cases": [
            {"input": ([["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], ["oath","pea","eat","rain"]), "expected": ["eat","oath"], "unordered": True},
            {"input": ([["a","b"],["c","d"]], ["abcb"]), "expected": []},
        ],
        "solution": """\
def find_words(board, words):
    trie, result = {}, []
    for word in words:
        node = trie
        for c in word: node = node.setdefault(c, {})
        node['#'] = word
    rows, cols = len(board), len(board[0])
    def dfs(r, c, node):
        ch = board[r][c]
        if ch not in node: return
        nxt = node[ch]
        if '#' in nxt:
            result.append(nxt['#'])
            del nxt['#']
        board[r][c] = '#'
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<rows and 0<=nc<cols:
                dfs(nr, nc, nxt)
        board[r][c] = ch
    for r in range(rows):
        for c in range(cols):
            dfs(r, c, trie)
    return result
""",
    },

    # ── Heap / Priority Queue ─────────────────────────────────────
    {
        "id": 62,
        "title": "Kth Largest Element In a Stream",
        "category": "Heap / Priority Queue",
        "difficulty": "Easy",
        "description": "Design a class that finds the kth largest element in a stream.",
        "python_tips": "Hint: maintain a min-heap of size k. The root is always the kth largest.",
        "starter_code": "class KthLargest:\n    def __init__(self, k, nums): pass\n    def add(self, val): pass\n",
        "test_cases": [
            {
                "class_test": True,
                "class_name": "KthLargest",
                "operations": ["KthLargest", "add", "add", "add", "add", "add"],
                "arguments": [[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]],
                "expected": [None, 4, 5, 5, 8, 8],
            },
        ],
        "solution": """\
class KthLargest:
    def __init__(self, k, nums):
        import heapq
        self.k, self.heap = k, nums
        heapq.heapify(self.heap)
        while len(self.heap) > k: heapq.heappop(self.heap)
    def add(self, val):
        import heapq
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k: heapq.heappop(self.heap)
        return self.heap[0]
""",
    },
    {
        "id": 63,
        "title": "Last Stone Weight",
        "category": "Heap / Priority Queue",
        "difficulty": "Easy",
        "description": "Stones are smashed together. The heaviest two are chosen each round. Return the weight of the last stone (or 0).",
        "python_tips": "Hint: use a max-heap (negate values since Python's heapq is a min-heap).",
        "starter_code": "def last_stone_weight(stones):\n    pass\n",
        "test_cases": [
            {"input": ([2,7,4,1,8,1],), "expected": 1},
            {"input": ([1],), "expected": 1},
        ],
        "solution": """\
def last_stone_weight(stones):
    import heapq
    heap = [-s for s in stones]
    heapq.heapify(heap)
    while len(heap) > 1:
        a = -heapq.heappop(heap)
        b = -heapq.heappop(heap)
        if a != b: heapq.heappush(heap, -(a - b))
    return -heap[0] if heap else 0
""",
    },
    {
        "id": 64,
        "title": "K Closest Points to Origin",
        "category": "Heap / Priority Queue",
        "difficulty": "Medium",
        "description": "Given a list of points, return the k closest to the origin (0,0).",
        "python_tips": "Hint: use heapq.nsmallest with key=lambda p: p[0]**2 + p[1]**2.",
        "starter_code": "def k_closest(points, k):\n    pass\n",
        "test_cases": [
            {"input": ([[1,3],[-2,2]], 1), "expected": [[-2,2]]},
            {"input": ([[3,3],[5,-1],[-2,4]], 2), "expected": [[3,3],[-2,4]], "unordered": True},
        ],
        "solution": """\
def k_closest(points, k):
    import heapq
    return heapq.nsmallest(k, points, key=lambda p: p[0]**2 + p[1]**2)
""",
    },
    {
        "id": 65,
        "title": "Task Scheduler",
        "category": "Heap / Priority Queue",
        "difficulty": "Medium",
        "description": "Given a list of CPU tasks and cooldown n, find the minimum intervals needed to finish all tasks.",
        "python_tips": "Hint: greedy — always pick the most frequent remaining task. Use a max-heap and a cooldown queue.",
        "starter_code": "def least_interval(tasks, n):\n    pass\n",
        "test_cases": [
            {"input": (["A","A","A","B","B","B"], 2), "expected": 8},
            {"input": (["A","C","A","B","D","B"], 1), "expected": 6},
        ],
        "solution": """\
def least_interval(tasks, n):
    import heapq
    from collections import Counter, deque
    count = Counter(tasks)
    heap = [-c for c in count.values()]
    heapq.heapify(heap)
    time, q = 0, deque()
    while heap or q:
        time += 1
        if heap:
            cnt = 1 + heapq.heappop(heap)
            if cnt: q.append((cnt, time + n))
        if q and q[0][1] == time:
            heapq.heappush(heap, q.popleft()[0])
    return time
""",
    },
    {
        "id": 66,
        "title": "Design Twitter",
        "category": "Heap / Priority Queue",
        "difficulty": "Medium",
        "description": "Design a simplified Twitter: postTweet, getNewsFeed (10 most recent from followed users), follow, unfollow.",
        "python_tips": "Hint: use a heap to merge each user's tweet list. Store tweets as (timestamp, tweetId).",
        "starter_code": "class Twitter:\n    def __init__(self): pass\n    def post_tweet(self, userId, tweetId): pass\n    def get_news_feed(self, userId): pass\n    def follow(self, followerId, followeeId): pass\n    def unfollow(self, followerId, followeeId): pass\n",
        "test_cases": [
            {
                "class_test": True,
                "class_name": "Twitter",
                "operations": ["Twitter", "post_tweet", "get_news_feed", "follow", "post_tweet", "get_news_feed"],
                "arguments":  [[], [1, 5], [1], [1, 2], [2, 6], [1]],
                "expected":   [None, None, [5], None, None, [6, 5]],
            },
        ],
        "solution": """\
class Twitter:
    def __init__(self):
        from collections import defaultdict
        import heapq
        self.count = 0
        self.tweets = defaultdict(list)
        self.following = defaultdict(set)
        self.heapq = heapq
    def post_tweet(self, userId, tweetId):
        self.tweets[userId].append((self.count, tweetId))
        self.count -= 1
    def get_news_feed(self, userId):
        heap = []
        self.following[userId].add(userId)
        for uid in self.following[userId]:
            if self.tweets[uid]:
                idx = len(self.tweets[uid]) - 1
                cnt, tid = self.tweets[uid][idx]
                self.heapq.heappush(heap, (cnt, tid, uid, idx - 1))
        feed = []
        while heap and len(feed) < 10:
            cnt, tid, uid, idx = self.heapq.heappop(heap)
            feed.append(tid)
            if idx >= 0:
                c2, t2 = self.tweets[uid][idx]
                self.heapq.heappush(heap, (c2, t2, uid, idx - 1))
        return feed
    def follow(self, followerId, followeeId):
        self.following[followerId].add(followeeId)
    def unfollow(self, followerId, followeeId):
        self.following[followerId].discard(followeeId)
""",
    },
    {
        "id": 67,
        "title": "Find Median From Data Stream",
        "category": "Heap / Priority Queue",
        "difficulty": "Hard",
        "description": "Design a data structure to add numbers and find the median at any time.",
        "python_tips": "Hint: use two heaps — a max-heap for the lower half, a min-heap for the upper half. Keep them balanced.",
        "starter_code": "class MedianFinder:\n    def __init__(self): pass\n    def add_num(self, num): pass\n    def find_median(self): pass\n",
        "test_cases": [
            {
                "class_test": True,
                "class_name": "MedianFinder",
                "operations": ["MedianFinder", "add_num", "add_num", "find_median", "add_num", "find_median"],
                "arguments": [[], [1], [2], [], [3], []],
                "expected": [None, None, None, 1.5, None, 2.0],
            },
        ],
        "solution": """\
class MedianFinder:
    def __init__(self):
        import heapq
        self.small = []  # max-heap (negated)
        self.large = []  # min-heap
        self.heapq = heapq
    def add_num(self, num):
        self.heapq.heappush(self.small, -num)
        if self.small and self.large and -self.small[0] > self.large[0]:
            self.heapq.heappush(self.large, -self.heapq.heappop(self.small))
        if len(self.small) > len(self.large) + 1:
            self.heapq.heappush(self.large, -self.heapq.heappop(self.small))
        if len(self.large) > len(self.small):
            self.heapq.heappush(self.small, -self.heapq.heappop(self.large))
    def find_median(self):
        if len(self.small) > len(self.large): return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
""",
    },

    # ── Backtracking ──────────────────────────────────────────────
    {
        "id": 68,
        "title": "Subsets",
        "category": "Backtracking",
        "difficulty": "Medium",
        "description": "Given an integer array with unique elements, return all possible subsets.",
        "python_tips": "Hint: backtracking — at each index either include or skip the element.",
        "starter_code": "def subsets(nums):\n    pass\n",
        "test_cases": [
            {"input": ([1,2,3],), "expected": [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]], "unordered": True},
        ],
        "solution": """\
def subsets(nums):
    result = []
    def bt(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            bt(i+1, path)
            path.pop()
    bt(0, [])
    return result
""",
    },
    {
        "id": 69,
        "title": "Combination Sum",
        "category": "Backtracking",
        "difficulty": "Medium",
        "description": "Find all unique combinations in candidates that sum to target. You may reuse the same number.",
        "python_tips": "Hint: backtracking — allow reusing same index. Stop when remaining sum < 0.",
        "starter_code": "def combination_sum(candidates, target):\n    pass\n",
        "test_cases": [
            {"input": ([2,3,6,7], 7), "expected": [[2,2,3],[7]], "unordered": True},
        ],
        "solution": """\
def combination_sum(candidates, target):
    result = []
    def bt(start, path, remaining):
        if remaining == 0: result.append(path[:]); return
        for i in range(start, len(candidates)):
            if candidates[i] <= remaining:
                path.append(candidates[i])
                bt(i, path, remaining - candidates[i])
                path.pop()
    bt(0, [], target)
    return result
""",
    },
    {
        "id": 70,
        "title": "Permutations",
        "category": "Backtracking",
        "difficulty": "Medium",
        "description": "Given a list of distinct integers, return all possible permutations.",
        "python_tips": "Hint: backtracking — swap elements in-place or track used elements with a boolean array.",
        "starter_code": "def permute(nums):\n    pass\n",
        "test_cases": [
            {"input": ([1,2,3],), "expected": [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]], "unordered": True},
        ],
        "solution": """\
def permute(nums):
    result = []
    def bt(path, remaining):
        if not remaining: result.append(path[:]); return
        for i in range(len(remaining)):
            path.append(remaining[i])
            bt(path, remaining[:i] + remaining[i+1:])
            path.pop()
    bt([], nums)
    return result
""",
    },
    {
        "id": 71,
        "title": "Subsets II",
        "category": "Backtracking",
        "difficulty": "Medium",
        "description": "Given an integer array that may contain duplicates, return all possible subsets (no duplicates).",
        "python_tips": "Hint: sort first, then skip duplicates at the same recursion depth.",
        "starter_code": "def subsets_with_dup(nums):\n    pass\n",
        "test_cases": [
            {"input": ([1,2,2],), "expected": [[],[1],[1,2],[1,2,2],[2],[2,2]], "unordered": True},
        ],
        "solution": """\
def subsets_with_dup(nums):
    nums.sort()
    result = []
    def bt(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i-1]: continue
            path.append(nums[i])
            bt(i+1, path)
            path.pop()
    bt(0, [])
    return result
""",
    },
    {
        "id": 72,
        "title": "Combination Sum II",
        "category": "Backtracking",
        "difficulty": "Medium",
        "description": "Given candidates (may have duplicates), find all unique combinations summing to target. Each number used once.",
        "python_tips": "Hint: sort + backtracking; skip duplicates at the same depth level.",
        "starter_code": "def combination_sum2(candidates, target):\n    pass\n",
        "test_cases": [
            {"input": ([10,1,2,7,6,1,5], 8), "expected": [[1,1,6],[1,2,5],[1,7],[2,6]], "unordered": True},
        ],
        "solution": """\
def combination_sum2(candidates, target):
    candidates.sort()
    result = []
    def bt(start, path, remaining):
        if remaining == 0: result.append(path[:]); return
        for i in range(start, len(candidates)):
            if i > start and candidates[i] == candidates[i-1]: continue
            if candidates[i] > remaining: break
            path.append(candidates[i])
            bt(i+1, path, remaining - candidates[i])
            path.pop()
    bt(0, [], target)
    return result
""",
    },
    {
        "id": 73,
        "title": "Word Search",
        "category": "Backtracking",
        "difficulty": "Medium",
        "description": "Given a 2D board and a word, return True if the word exists in the grid (adjacent cells, no reuse).",
        "python_tips": "Hint: DFS/backtracking from each cell. Mark cells as visited by temporarily replacing the character.",
        "starter_code": "def exist(board, word):\n    pass\n",
        "test_cases": [
            {"input": ([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCCED"), "expected": True},
            {"input": ([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "SEE"),    "expected": True},
            {"input": ([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCB"),   "expected": False},
        ],
        "solution": """\
def exist(board, word):
    rows, cols = len(board), len(board[0])
    def dfs(r, c, i):
        if i == len(word): return True
        if r<0 or r>=rows or c<0 or c>=cols or board[r][c] != word[i]: return False
        board[r][c] = '#'
        found = any(dfs(r+dr, c+dc, i+1) for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)])
        board[r][c] = word[i]
        return found
    return any(dfs(r,c,0) for r in range(rows) for c in range(cols))
""",
    },
    {
        "id": 74,
        "title": "Palindrome Partitioning",
        "category": "Backtracking",
        "difficulty": "Medium",
        "description": "Partition a string such that every substring is a palindrome. Return all possible partitions.",
        "python_tips": "Hint: backtracking — at each position, try all prefixes that are palindromes.",
        "starter_code": "def partition(s):\n    pass\n",
        "test_cases": [
            {"input": ("aab",), "expected": [["a","a","b"],["aa","b"]], "unordered": True},
        ],
        "solution": """\
def partition(s):
    result = []
    def is_pal(t): return t == t[::-1]
    def bt(start, path):
        if start == len(s): result.append(path[:]); return
        for end in range(start+1, len(s)+1):
            if is_pal(s[start:end]):
                path.append(s[start:end])
                bt(end, path)
                path.pop()
    bt(0, [])
    return result
""",
    },
    {
        "id": 75,
        "title": "Letter Combinations of a Phone Number",
        "category": "Backtracking",
        "difficulty": "Medium",
        "description": "Given a string of digits 2-9, return all possible letter combinations from a phone keypad.",
        "python_tips": "Hint: use a digit→letters map, then backtracking one digit at a time.",
        "starter_code": "def letter_combinations(digits):\n    pass\n",
        "test_cases": [
            {"input": ("23",), "expected": ["ad","ae","af","bd","be","bf","cd","ce","cf"], "unordered": True},
            {"input": ("",),   "expected": []},
        ],
        "solution": """\
def letter_combinations(digits):
    if not digits: return []
    phone = {'2':'abc','3':'def','4':'ghi','5':'jkl',
             '6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
    result = []
    def bt(i, path):
        if i == len(digits): result.append(''.join(path)); return
        for c in phone[digits[i]]:
            path.append(c)
            bt(i+1, path)
            path.pop()
    bt(0, [])
    return result
""",
    },

    # ─────────────────────────────────────────────────────────────
    # GRAPHS
    # ─────────────────────────────────────────────────────────────
    {
        "id": 76,
        "title": "Number of Islands",
        "category": "Graphs",
        "difficulty": "Medium",
        "description": """\
Given an `m x n` 2D grid of characters `'1'` (land) and `'0'` (water), return the **number of islands**.

An **island** is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are surrounded by water.

**Example 1:**
```
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
```

**Example 2:**
```
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
```

**Constraints:**
- m == len(grid)
- n == len(grid[i])
- 1 <= m, n <= 300
- grid[i][j] is '0' or '1'
""",
        "python_tips": """\
**Key concept: Depth-First Search (DFS) on a grid**

Think of the grid as a graph where each cell is a node connected to its 4 neighbors (up, down, left, right).

**The core idea:**
1. Loop through every cell in the grid.
2. When you find a `'1'` you haven't visited, you've found a new island — increment your count.
3. Use DFS (or BFS) to "sink" the entire island by marking all connected `'1'`s as `'0'` (visited).
4. This way, you won't count the same island twice.

**DFS pattern on a grid:**
```python3
def dfs(grid, r, c):
    if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]):
        return
    if grid[r][c] == '0':
        return
    grid[r][c] = '0'  # mark visited
    dfs(grid, r+1, c)
    dfs(grid, r-1, c)
    dfs(grid, r, c+1)
    dfs(grid, r, c-1)
```

**Time:** O(m * n) — visit every cell at most once.
""",
        "starter_code": """\
def num_islands(grid):
    \"\"\"
    Args:
        grid (list[list[str]]): 2D grid of '1' (land) and '0' (water)

    Returns:
        int: number of islands
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]],), "expected": 1},
            {"input": ([["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]],), "expected": 3},
            {"input": ([["1","0","1"],["0","0","0"],["1","0","1"]],), "expected": 4},
        ],
        "solution": """\
def num_islands(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return
        grid[r][c] = '0'
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)
    return count
""",
    },

    {
        "id": 77,
        "title": "Clone Graph",
        "category": "Graphs",
        "difficulty": "Medium",
        "description": """\
Given a reference of a node in a connected undirected graph, return a **deep copy** (clone) of the graph.

For simplicity, the graph is represented as an **adjacency list** where `adj[i]` is a list of neighbors of node `i+1` (1-indexed node values).

Your function should return the same adjacency list structure (a deep copy).

**Example 1:**
```
Input:  adj = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]
```
This represents a graph with 4 nodes:
- Node 1 connects to nodes 2 and 4
- Node 2 connects to nodes 1 and 3
- Node 3 connects to nodes 2 and 4
- Node 4 connects to nodes 1 and 3

**Example 2:**
```
Input:  adj = [[]]
Output: [[]]
```

**Constraints:**
- 1 <= number of nodes <= 100
- Node values are unique and in range [1, number of nodes]
- No duplicate edges, no self-loops
""",
        "python_tips": """\
**Key concept: Graph traversal with a visited dictionary**

When cloning a graph, the tricky part is handling cycles — you might visit the same node twice. A dictionary maps original nodes to their clones so you never duplicate.

**The core idea (BFS/DFS):**
1. Start from the first node. Create its clone and store it in a `visited` dict.
2. For each neighbor, if not yet cloned, clone it and add to the queue/stack.
3. Connect the cloned node to its cloned neighbors.

**With adjacency list representation**, this simplifies to a deep copy:
```python3
import copy
clone = copy.deepcopy(adj)
```

But to practice the graph traversal pattern, implement BFS yourself:
```python3
from collections import deque
visited = {}
queue = deque([start_node])
```

**Time:** O(V + E) where V = nodes, E = edges.
""",
        "starter_code": """\
def clone_graph(adj):
    \"\"\"
    Args:
        adj (list[list[int]]): adjacency list (1-indexed node values)

    Returns:
        list[list[int]]: deep copy of the adjacency list
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([[2,4],[1,3],[2,4],[1,3]],), "expected": [[2,4],[1,3],[2,4],[1,3]]},
            {"input": ([[2],[1]],), "expected": [[2],[1]]},
            {"input": ([[]],), "expected": [[]]},
        ],
        "solution": """\
def clone_graph(adj):
    if not adj:
        return []
    from collections import deque
    n = len(adj)
    clone = [None] * n
    visited = [False] * n
    queue = deque([0])
    visited[0] = True
    clone[0] = []
    while queue:
        node = queue.popleft()
        clone[node] = list(adj[node])  # deep copy the neighbor list
        for neighbor_val in adj[node]:
            idx = neighbor_val - 1  # convert 1-indexed to 0-indexed
            if not visited[idx]:
                visited[idx] = True
                queue.append(idx)
    return clone
""",
    },

    {
        "id": 78,
        "title": "Pacific Atlantic Water Flow",
        "category": "Graphs",
        "difficulty": "Medium",
        "description": """\
There is an `m x n` rectangular island that borders both the **Pacific Ocean** and **Atlantic Ocean**. The Pacific ocean touches the island's left and top edges, and the Atlantic ocean touches the island's right and bottom edges.

The island receives a lot of rain, and the rain water can flow to neighboring cells (up, down, left, right) if the neighboring cell's height is **less than or equal to** the current cell's height. Water can flow from any cell adjacent to an ocean into the ocean.

Return a list of grid coordinates `[r, c]` where rain water can flow to **both** the Pacific and Atlantic oceans.

**Example 1:**
```
Input: heights = [
  [1,2,2,3,5],
  [3,2,3,4,4],
  [2,4,5,3,1],
  [6,7,1,4,5],
  [5,1,1,2,4]
]
Output: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
```

**Example 2:**
```
Input: heights = [[1]]
Output: [[0,0]]
```

**Constraints:**
- m == len(heights), n == len(heights[0])
- 1 <= m, n <= 200
- 0 <= heights[r][c] <= 10⁵
""",
        "python_tips": """\
**Key concept: Reverse DFS from ocean borders**

Instead of flowing water *from* each cell *to* the ocean (expensive), do it backwards:
1. Start DFS/BFS from all **Pacific border** cells (top row + left column). Mark all reachable cells.
2. Start DFS/BFS from all **Atlantic border** cells (bottom row + right column). Mark all reachable cells.
3. Any cell in **both** sets can reach both oceans.

**Why reverse?** When flowing backwards, water flows from lower to higher (or equal), so the condition is `heights[nr][nc] >= heights[r][c]`.

```python3
def dfs(r, c, visited, prev_height):
    if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols:
        return
    if heights[r][c] < prev_height:
        return
    visited.add((r, c))
    for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
        dfs(r+dr, c+dc, visited, heights[r][c])
```

**Time:** O(m * n) — each cell visited at most twice (once per ocean).
""",
        "starter_code": """\
def pacific_atlantic(heights):
    \"\"\"
    Args:
        heights (list[list[int]]): m x n grid of heights

    Returns:
        list[list[int]]: coordinates [r, c] that can reach both oceans
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]],), "expected": [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]], "unordered": True},
            {"input": ([[1]],), "expected": [[0,0]], "unordered": True},
        ],
        "solution": """\
def pacific_atlantic(heights):
    if not heights:
        return []
    rows, cols = len(heights), len(heights[0])
    pacific = set()
    atlantic = set()

    def dfs(r, c, visited, prev_height):
        if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols:
            return
        if heights[r][c] < prev_height:
            return
        visited.add((r, c))
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            dfs(r+dr, c+dc, visited, heights[r][c])

    for c in range(cols):
        dfs(0, c, pacific, heights[0][c])
        dfs(rows-1, c, atlantic, heights[rows-1][c])
    for r in range(rows):
        dfs(r, 0, pacific, heights[r][0])
        dfs(r, cols-1, atlantic, heights[r][cols-1])

    return [[r, c] for r, c in pacific & atlantic]
""",
    },

    {
        "id": 79,
        "title": "Course Schedule",
        "category": "Graphs",
        "difficulty": "Medium",
        "description": """\
There are a total of `numCourses` courses you have to take, labeled from `0` to `numCourses - 1`. You are given an array `prerequisites` where `prerequisites[i] = [a, b]` means you **must** take course `b` before course `a`.

Return `True` if you can finish all courses, or `False` if there is a circular dependency.

**Example 1:**
```
Input:  numCourses = 2, prerequisites = [[1,0]]
Output: True
Explanation: Take course 0 first, then course 1.
```

**Example 2:**
```
Input:  numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: False
Explanation: Course 0 requires 1, and course 1 requires 0 — a cycle!
```

**Example 3:**
```
Input:  numCourses = 4, prerequisites = [[1,0],[2,1],[3,2]]
Output: True
```

**Constraints:**
- 1 <= numCourses <= 2000
- 0 <= len(prerequisites) <= 5000
- prerequisites[i].length == 2
- All prerequisite pairs are unique
""",
        "python_tips": """\
**Key concept: Cycle detection in a directed graph (topological sort)**

This problem asks: "Does the prerequisite graph have a cycle?" If yes, you can't finish all courses.

**Approach 1 — DFS with 3-color marking:**
- WHITE (unvisited), GRAY (in current DFS path), BLACK (fully processed)
- If you visit a GRAY node during DFS → cycle detected!

```python3
# 0 = unvisited, 1 = in-progress, 2 = done
def dfs(course):
    if state[course] == 1: return False  # cycle!
    if state[course] == 2: return True   # already done
    state[course] = 1
    for prereq in graph[course]:
        if not dfs(prereq): return False
    state[course] = 2
    return True
```

**Approach 2 — BFS (Kahn's algorithm):**
- Track in-degrees. Start with courses that have no prerequisites.
- Remove them and reduce neighbors' in-degrees. If all courses get removed, no cycle.

**Time:** O(V + E) where V = courses, E = prerequisites.
""",
        "starter_code": """\
def can_finish(num_courses, prerequisites):
    \"\"\"
    Args:
        num_courses   (int):             total number of courses
        prerequisites (list[list[int]]): pairs [a, b] meaning b must come before a

    Returns:
        bool: True if all courses can be finished
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": (2, [[1,0]]),         "expected": True},
            {"input": (2, [[1,0],[0,1]]),   "expected": False},
            {"input": (4, [[1,0],[2,1],[3,2]]), "expected": True},
        ],
        "solution": """\
def can_finish(num_courses, prerequisites):
    graph = [[] for _ in range(num_courses)]
    for a, b in prerequisites:
        graph[a].append(b)

    # 0=unvisited, 1=in-progress, 2=done
    state = [0] * num_courses

    def dfs(course):
        if state[course] == 1:
            return False
        if state[course] == 2:
            return True
        state[course] = 1
        for pre in graph[course]:
            if not dfs(pre):
                return False
        state[course] = 2
        return True

    for c in range(num_courses):
        if not dfs(c):
            return False
    return True
""",
    },

    {
        "id": 80,
        "title": "Number of Connected Components in an Undirected Graph",
        "category": "Graphs",
        "difficulty": "Medium",
        "description": """\
You have `n` nodes labeled from `0` to `n - 1` and a list of undirected `edges` (where `edges[i] = [a, b]` means there is an edge between nodes `a` and `b`).

Return the **number of connected components** in the graph.

**Example 1:**
```
Input:  n = 5, edges = [[0,1],[1,2],[3,4]]
Output: 2
Explanation: Components are {0,1,2} and {3,4}.
```

**Example 2:**
```
Input:  n = 5, edges = [[0,1],[1,2],[2,3],[3,4]]
Output: 1
Explanation: All nodes are connected.
```

**Example 3:**
```
Input:  n = 4, edges = []
Output: 4
Explanation: No edges, so each node is its own component.
```

**Constraints:**
- 1 <= n <= 2000
- 0 <= len(edges) <= 5000
- edges[i].length == 2
- No duplicate edges
""",
        "python_tips": """\
**Key concept: Union-Find (Disjoint Set Union) or DFS**

**Approach 1 — DFS/BFS:**
Build an adjacency list, then count how many times you start a new DFS (each start = new component).

**Approach 2 — Union-Find (great to learn!):**
```python3
parent = list(range(n))  # each node is its own parent

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # path compression
        x = parent[x]
    return x

def union(a, b):
    pa, pb = find(a), find(b)
    if pa != pb:
        parent[pa] = pb
```
Start with `n` components. Each successful `union` reduces count by 1.

**Time:** O(V + E) for DFS; nearly O(1) per operation for Union-Find with path compression.
""",
        "starter_code": """\
def count_components(n, edges):
    \"\"\"
    Args:
        n     (int):             number of nodes (0 to n-1)
        edges (list[list[int]]): undirected edges

    Returns:
        int: number of connected components
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": (5, [[0,1],[1,2],[3,4]]),         "expected": 2},
            {"input": (5, [[0,1],[1,2],[2,3],[3,4]]),   "expected": 1},
            {"input": (4, []),                           "expected": 4},
        ],
        "solution": """\
def count_components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb
            return True
        return False

    components = n
    for a, b in edges:
        if union(a, b):
            components -= 1
    return components
""",
    },

    {
        "id": 81,
        "title": "Graph Valid Tree",
        "category": "Graphs",
        "difficulty": "Medium",
        "description": """\
Given `n` nodes labeled from `0` to `n - 1` and a list of undirected `edges`, check whether these edges form a **valid tree**.

A valid tree must satisfy:
1. **Connected** — all nodes are reachable from any other node.
2. **No cycles** — there is exactly one path between any two nodes.

**Example 1:**
```
Input:  n = 5, edges = [[0,1],[0,2],[0,3],[1,4]]
Output: True
```

**Example 2:**
```
Input:  n = 5, edges = [[0,1],[1,2],[2,3],[1,3],[1,4]]
Output: False
Explanation: There's a cycle: 1 -> 2 -> 3 -> 1
```

**Example 3:**
```
Input:  n = 4, edges = [[0,1],[2,3]]
Output: False
Explanation: Not connected — {0,1} and {2,3} are separate.
```

**Constraints:**
- 1 <= n <= 2000
- 0 <= len(edges) <= 5000
- No duplicate edges, no self-loops
""",
        "python_tips": """\
**Key insight: A graph is a valid tree if and only if:**
1. It has exactly `n - 1` edges, AND
2. It is fully connected (one component).

If edges != n - 1, return False immediately (too few = disconnected, too many = cycle).

**Using Union-Find:**
```python3
parent = list(range(n))
for a, b in edges:
    if find(a) == find(b):  # already connected → cycle!
        return False
    union(a, b)
# Check: exactly one component
```

Or simply: if `len(edges) == n - 1` and Union-Find produces 1 component, it's a tree.

**Using DFS:** Start from node 0, track parent to avoid going back. If you visit an already-visited node, there's a cycle.

**Time:** O(V + E).
""",
        "starter_code": """\
def valid_tree(n, edges):
    \"\"\"
    Args:
        n     (int):             number of nodes (0 to n-1)
        edges (list[list[int]]): undirected edges

    Returns:
        bool: True if edges form a valid tree
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": (5, [[0,1],[0,2],[0,3],[1,4]]),         "expected": True},
            {"input": (5, [[0,1],[1,2],[2,3],[1,3],[1,4]]),   "expected": False},
            {"input": (4, [[0,1],[2,3]]),                      "expected": False},
        ],
        "solution": """\
def valid_tree(n, edges):
    if len(edges) != n - 1:
        return False

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa == pb:
            return False
        parent[pa] = pb
        return True

    for a, b in edges:
        if not union(a, b):
            return False
    return True
""",
    },

    # ─────────────────────────────────────────────────────────────
    # ADVANCED GRAPHS
    # ─────────────────────────────────────────────────────────────
    {
        "id": 82,
        "title": "Alien Dictionary",
        "category": "Advanced Graphs",
        "difficulty": "Hard",
        "description": """\
There is a new alien language that uses the English alphabet. However, the order of the letters is unknown to you.

You are given a list of strings `words` from the alien language's dictionary, where the strings are **sorted lexicographically** by the rules of this new language.

Derive the order of letters in this language and return it as a string. If the order is invalid (i.e., no valid ordering exists), return an empty string `""`. If there are multiple valid orderings, return **any** of them.

**Example 1:**
```
Input:  words = ["wrt","wrf","er","ett","rftt"]
Output: "wertf"
Explanation:
  wrt vs wrf → t before f
  wrf vs er  → w before e
  er vs ett  → r before t
  ett vs rftt → e before r
  So: w → e → r → t → f
```

**Example 2:**
```
Input:  words = ["z","x"]
Output: "zx"
```

**Example 3:**
```
Input:  words = ["z","x","z"]
Output: ""
Explanation: The order is invalid (z before x and x before z).
```

**Constraints:**
- 1 <= len(words) <= 100
- 1 <= len(words[i]) <= 100
- words[i] consists of only lowercase English letters
""",
        "python_tips": """\
**Key concept: Topological sort on character ordering**

**Step 1 — Build a directed graph of character orderings:**
Compare each pair of adjacent words. Find the first differing character — that gives you an edge (char_a comes before char_b).

**Important edge case:** If a longer word comes before its prefix (e.g., ["abc", "ab"]), the ordering is **invalid**.

**Step 2 — Topological sort (BFS/Kahn's algorithm):**
```python3
from collections import deque, defaultdict

in_degree = {c: 0 for c in all_chars}
graph = defaultdict(set)

# Build edges from adjacent word comparisons
# Then BFS starting from chars with in_degree 0
queue = deque([c for c in in_degree if in_degree[c] == 0])
result = []
while queue:
    c = queue.popleft()
    result.append(c)
    for neighbor in graph[c]:
        in_degree[neighbor] -= 1
        if in_degree[neighbor] == 0:
            queue.append(neighbor)
```

If `len(result) < len(all_chars)`, there's a cycle → return "".

**Time:** O(C) where C = total characters across all words.
""",
        "starter_code": """\
def alien_order(words):
    \"\"\"
    Args:
        words (list[str]): words sorted in alien dictionary order

    Returns:
        str: characters in alien alphabetical order, or \"\" if invalid
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": (["wrt","wrf","er","ett","rftt"],), "expected": "wertf"},
            {"input": (["z","x"],),                        "expected": "zx"},
            {"input": (["z","x","z"],),                    "expected": ""},
        ],
        "solution": """\
def alien_order(words):
    from collections import defaultdict, deque

    # Collect all unique characters
    in_degree = {c: 0 for word in words for c in word}
    graph = defaultdict(set)

    # Build graph from adjacent word pairs
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i+1]
        min_len = min(len(w1), len(w2))
        # Invalid: longer word is prefix of shorter
        if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
            return ""
        for j in range(min_len):
            if w1[j] != w2[j]:
                if w2[j] not in graph[w1[j]]:
                    graph[w1[j]].add(w2[j])
                    in_degree[w2[j]] += 1
                break

    # BFS topological sort
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    result = []
    while queue:
        c = queue.popleft()
        result.append(c)
        for neighbor in graph[c]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) < len(in_degree):
        return ""
    return "".join(result)
""",
    },

    # ─────────────────────────────────────────────────────────────
    # DYNAMIC PROGRAMMING (1D)
    # ─────────────────────────────────────────────────────────────
    {
        "id": 83,
        "title": "Climbing Stairs",
        "category": "Dynamic Programming",
        "difficulty": "Easy",
        "description": """\
You are climbing a staircase. It takes `n` steps to reach the top. Each time you can either climb **1** or **2** steps.

In how many **distinct ways** can you climb to the top?

**Example 1:**
```
Input:  n = 2
Output: 2
Explanation: Two ways — (1+1) or (2).
```

**Example 2:**
```
Input:  n = 3
Output: 3
Explanation: Three ways — (1+1+1), (1+2), (2+1).
```

**Example 3:**
```
Input:  n = 5
Output: 8
```

**Constraints:**
- 1 <= n <= 45
""",
        "python_tips": """\
**Key concept: Dynamic Programming (Fibonacci pattern)**

This is the classic intro to DP! Notice the pattern:
- To reach step `n`, you either came from step `n-1` (took 1 step) or step `n-2` (took 2 steps).
- So: `ways(n) = ways(n-1) + ways(n-2)` — that's the Fibonacci sequence!

**Base cases:** `ways(1) = 1`, `ways(2) = 2`.

**Bottom-up approach (no recursion, O(1) space):**
```python3
def climb_stairs(n):
    if n <= 2:
        return n
    a, b = 1, 2  # ways(1), ways(2)
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b
```

You only need the last two values — no need for an entire array!

**Time:** O(n) | **Space:** O(1)
""",
        "starter_code": """\
def climb_stairs(n):
    \"\"\"
    Args:
        n (int): number of steps

    Returns:
        int: number of distinct ways to climb to the top
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": (2,), "expected": 2},
            {"input": (3,), "expected": 3},
            {"input": (5,), "expected": 8},
        ],
        "solution": """\
def climb_stairs(n):
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b
""",
    },

    {
        "id": 84,
        "title": "House Robber",
        "category": "Dynamic Programming",
        "difficulty": "Medium",
        "description": """\
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. The only constraint is that **adjacent houses have security systems connected** — if two adjacent houses are broken into on the same night, the police will be alerted.

Given an integer array `nums` representing the amount of money at each house, return the **maximum** amount you can rob **without alerting the police**.

**Example 1:**
```
Input:  nums = [1,2,3,1]
Output: 4
Explanation: Rob house 0 (1) + house 2 (3) = 4.
```

**Example 2:**
```
Input:  nums = [2,7,9,3,1]
Output: 12
Explanation: Rob house 0 (2) + house 2 (9) + house 4 (1) = 12.
```

**Example 3:**
```
Input:  nums = [2,1,1,2]
Output: 4
Explanation: Rob house 0 (2) + house 3 (2) = 4.
```

**Constraints:**
- 1 <= len(nums) <= 100
- 0 <= nums[i] <= 400
""",
        "python_tips": """\
**Key concept: DP with "take or skip" decision**

At each house, you have two choices:
1. **Rob it** — add its value to the best total from two houses back (skip the adjacent one).
2. **Skip it** — keep the best total from the previous house.

**Recurrence:** `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`

**Base cases:** `dp[0] = nums[0]`, `dp[1] = max(nums[0], nums[1])`

**Optimized O(1) space:**
```python3
def rob(nums):
    prev2, prev1 = 0, 0
    for num in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + num)
    return prev1
```

Think of `prev1` as "best so far including or excluding the last house" and `prev2` as "best from two houses ago."

**Time:** O(n) | **Space:** O(1)
""",
        "starter_code": """\
def rob(nums):
    \"\"\"
    Args:
        nums (list[int]): money at each house

    Returns:
        int: maximum money you can rob without robbing adjacent houses
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([1,2,3,1],),   "expected": 4},
            {"input": ([2,7,9,3,1],), "expected": 12},
            {"input": ([2,1,1,2],),   "expected": 4},
        ],
        "solution": """\
def rob(nums):
    prev2, prev1 = 0, 0
    for num in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + num)
    return prev1
""",
    },

    {
        "id": 85,
        "title": "House Robber II",
        "category": "Dynamic Programming",
        "difficulty": "Medium",
        "description": """\
You are a robber, and all houses are arranged **in a circle**. That means the first house and the last house are **adjacent**. You cannot rob two adjacent houses.

Given an integer array `nums` representing money at each house, return the **maximum** amount you can rob.

**Example 1:**
```
Input:  nums = [2,3,2]
Output: 3
Explanation: You can't rob house 0 and house 2 (they're adjacent in a circle).
             Best is to rob house 1 = 3.
```

**Example 2:**
```
Input:  nums = [1,2,3,1]
Output: 4
Explanation: Rob house 0 (1) + house 2 (3) = 4.
```

**Example 3:**
```
Input:  nums = [1,2,3]
Output: 3
```

**Constraints:**
- 1 <= len(nums) <= 100
- 0 <= nums[i] <= 1000
""",
        "python_tips": """\
**Key insight: Reduce circular to linear!**

Since house 0 and house n-1 are adjacent in a circle, you can never rob both. So split into two cases:
1. Rob from house `0` to house `n-2` (exclude last).
2. Rob from house `1` to house `n-1` (exclude first).

Take the **max** of both cases. Each sub-problem is just the regular House Robber!

```python3
def rob_ii(nums):
    if len(nums) == 1:
        return nums[0]
    return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))

def rob_linear(nums):
    prev2, prev1 = 0, 0
    for num in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + num)
    return prev1
```

**Time:** O(n) | **Space:** O(1)
""",
        "starter_code": """\
def rob_ii(nums):
    \"\"\"
    Args:
        nums (list[int]): money at each house (arranged in a circle)

    Returns:
        int: maximum money you can rob
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([2,3,2],),   "expected": 3},
            {"input": ([1,2,3,1],), "expected": 4},
            {"input": ([1,2,3],),   "expected": 3},
        ],
        "solution": """\
def rob_ii(nums):
    if len(nums) == 1:
        return nums[0]

    def rob_linear(houses):
        prev2, prev1 = 0, 0
        for h in houses:
            prev2, prev1 = prev1, max(prev1, prev2 + h)
        return prev1

    return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))
""",
    },

    {
        "id": 86,
        "title": "Decode Ways",
        "category": "Dynamic Programming",
        "difficulty": "Medium",
        "description": """\
A message containing letters from A-Z can be **encoded** into numbers using the mapping:
```
'A' -> "1", 'B' -> "2", ..., 'Z' -> "26"
```

Given a string `s` containing only digits, return the **number of ways** to decode it.

Note that groupings like `"06"` are **not valid** (because "06" is not a valid encoding — leading zeros are not allowed).

**Example 1:**
```
Input:  s = "12"
Output: 2
Explanation: "12" can be decoded as "AB" (1 2) or "L" (12).
```

**Example 2:**
```
Input:  s = "226"
Output: 3
Explanation: "226" can be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
```

**Example 3:**
```
Input:  s = "06"
Output: 0
Explanation: "06" has no valid mapping (leading zero). There is no way to decode this.
```

**Constraints:**
- 1 <= len(s) <= 100
- s contains only digits and may contain leading zeros
""",
        "python_tips": """\
**Key concept: DP with 1-digit and 2-digit checks**

At each position `i`, you can decode:
1. **One digit** `s[i]` — valid if it's `'1'`..`'9'` (not `'0'`). Add `dp[i-1]` ways.
2. **Two digits** `s[i-1:i+1]` — valid if it's `"10"`..`"26"`. Add `dp[i-2]` ways.

**Recurrence:**
```python3
dp[i] = 0
if s[i] != '0':        dp[i] += dp[i-1]   # single digit
if 10 <= int(s[i-1:i+1]) <= 26:  dp[i] += dp[i-2]   # two digits
```

**Base case:** `dp[0] = 1` if `s[0] != '0'`, else `0`.

**O(1) space optimization:** You only need the last two DP values, so use two variables instead of an array.

**Watch out for zeros!** `'0'` by itself is invalid, so `s = "10"` has only 1 decoding ("J"), not 2.

**Time:** O(n) | **Space:** O(1)
""",
        "starter_code": """\
def num_decodings(s):
    \"\"\"
    Args:
        s (str): string of digits

    Returns:
        int: number of ways to decode the string
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ("12",),  "expected": 2},
            {"input": ("226",), "expected": 3},
            {"input": ("06",),  "expected": 0},
        ],
        "solution": """\
def num_decodings(s):
    if not s or s[0] == '0':
        return 0
    n = len(s)
    # prev2 = dp[i-2], prev1 = dp[i-1]
    prev2, prev1 = 1, 1
    for i in range(1, n):
        current = 0
        if s[i] != '0':
            current += prev1
        two_digit = int(s[i-1:i+1])
        if 10 <= two_digit <= 26:
            current += prev2
        prev2, prev1 = prev1, current
    return prev1
""",
    },

    # ─────────────────────────────────────────────────────────────
    # 1D DYNAMIC PROGRAMMING (continued)
    # ─────────────────────────────────────────────────────────────
    {
        "id": 87,
        "title": "Palindromic Substrings",
        "category": "Dynamic Programming",
        "difficulty": "Medium",
        "description": """\
Given a string `s`, return the **number of palindromic substrings** in it.

A substring is a contiguous sequence of characters within the string. A string is a palindrome when it reads the same backward as forward.

**Example 1:**
```
Input:  s = "abc"
Output: 3
Explanation: "a", "b", "c" — each single character is a palindrome.
```

**Example 2:**
```
Input:  s = "aaa"
Output: 6
Explanation: "a", "a", "a", "aa", "aa", "aaa" — six palindromic substrings.
```

**Constraints:**
- 1 ≤ len(s) ≤ 1000
- s consists of lowercase English letters.
""",
        "python_tips": """\
**Key Python concept: Expand Around Center**

Every palindrome has a center. For odd-length palindromes the center is one character; for even-length it is between two characters.

**The core idea:**
- For each index `i`, try expanding outward from two starting points:
  1. `(i, i)` — odd-length palindromes like "aba"
  2. `(i, i+1)` — even-length palindromes like "abba"
- While `s[left] == s[right]`, it is a palindrome — count it and keep expanding.

**Useful pattern:**
```python3
def count_from(s, left, right):
    count = 0
    while left >= 0 and right < len(s) and s[left] == s[right]:
        count += 1
        left -= 1
        right += 1
    return count
```

**Time complexity:** O(n²) — expand from each of n centers.
**Space complexity:** O(1) — no extra data structures.
""",
        "starter_code": """\
def count_substrings(s):
    \"\"\"
    Args:
        s (str): input string

    Returns:
        int: number of palindromic substrings
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ("abc",), "expected": 3},
            {"input": ("aaa",), "expected": 6},
            {"input": ("racecar",), "expected": 10},
        ],
        "solution": """\
def count_substrings(s):
    count = 0
    def expand(left, right):
        nonlocal count
        while left >= 0 and right < len(s) and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
    for i in range(len(s)):
        expand(i, i)      # odd-length
        expand(i, i + 1)  # even-length
    return count
""",
    },

    {
        "id": 88,
        "title": "Longest Palindromic Substring",
        "category": "Dynamic Programming",
        "difficulty": "Medium",
        "description": """\
Given a string `s`, return the **longest palindromic substring** in `s`.

If there are multiple answers of the same length, return any one of them.

**Example 1:**
```
Input:  s = "babad"
Output: "bab"   # "aba" is also valid
```

**Example 2:**
```
Input:  s = "cbbd"
Output: "bb"
```

**Example 3:**
```
Input:  s = "a"
Output: "a"
```

**Constraints:**
- 1 ≤ len(s) ≤ 1000
- s consists of only digits and English letters.
""",
        "python_tips": """\
**Key Python concept: Expand Around Center (tracking the best)**

This uses the same "expand around center" technique as Palindromic Substrings, but instead of counting, you track which palindrome is the longest.

**The core idea:**
- For each center, expand outward while `s[left] == s[right]`.
- If the current palindrome is longer than the best so far, update your answer.
- Try both odd-length centers `(i, i)` and even-length centers `(i, i+1)`.

**Useful Python slice:**
```python3
s[left:right+1]   # substring from left to right inclusive
```

**Time complexity:** O(n²) — expand from each of n centers.
**Space complexity:** O(1) — only storing indices.
""",
        "starter_code": """\
def longest_palindrome(s):
    \"\"\"
    Args:
        s (str): input string

    Returns:
        str: the longest palindromic substring
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ("babad",), "expected": "bab"},
            {"input": ("cbbd",), "expected": "bb"},
            {"input": ("a",), "expected": "a"},
        ],
        "solution": """\
def longest_palindrome(s):
    start, max_len = 0, 1
    def expand(left, right):
        nonlocal start, max_len
        while left >= 0 and right < len(s) and s[left] == s[right]:
            if right - left + 1 > max_len:
                start = left
                max_len = right - left + 1
            left -= 1
            right += 1
    for i in range(len(s)):
        expand(i, i)
        expand(i, i + 1)
    return s[start:start + max_len]
""",
    },

    {
        "id": 89,
        "title": "Coin Change",
        "category": "Dynamic Programming",
        "difficulty": "Medium",
        "description": """\
You are given an integer array `coins` representing coin denominations and an integer `amount` representing a total amount of money.

Return the **fewest number of coins** needed to make up that amount. If that amount cannot be made up by any combination of the coins, return `-1`.

You may assume you have an **infinite supply** of each coin.

**Example 1:**
```
Input:  coins = [1, 5, 10], amount = 12
Output: 3   # 10 + 1 + 1 = 12
```

**Example 2:**
```
Input:  coins = [2], amount = 3
Output: -1  # impossible with only 2-cent coins
```

**Example 3:**
```
Input:  coins = [1], amount = 0
Output: 0   # no coins needed for amount 0
```

**Constraints:**
- 1 ≤ len(coins) ≤ 12
- 1 ≤ coins[i] ≤ 2³¹ - 1
- 0 ≤ amount ≤ 10⁴
""",
        "python_tips": """\
**Key Python concept: Bottom-up Dynamic Programming**

Build up solutions from smaller amounts to the target amount.

**The core idea:**
- Create an array `dp` where `dp[i]` = fewest coins needed to make amount `i`.
- Initialize `dp[0] = 0` (zero coins for zero amount) and all others to infinity.
- For each amount from 1 to target, try every coin: if `coin ≤ i`, then `dp[i] = min(dp[i], dp[i - coin] + 1)`.

**Useful Python syntax:**
```python3
dp = [float('inf')] * (amount + 1)   # list filled with infinity
dp[0] = 0                             # base case
min(a, b)                             # return smaller value
```

**Time complexity:** O(amount * len(coins)).
**Space complexity:** O(amount).
""",
        "starter_code": """\
def coin_change(coins, amount):
    \"\"\"
    Args:
        coins  (list[int]): coin denominations
        amount (int):       target amount

    Returns:
        int: fewest coins needed, or -1 if impossible
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([1, 5, 10], 12), "expected": 3},
            {"input": ([2], 3), "expected": -1},
            {"input": ([1], 0), "expected": 0},
        ],
        "solution": """\
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
""",
    },

    {
        "id": 90,
        "title": "Word Break",
        "category": "Dynamic Programming",
        "difficulty": "Medium",
        "description": """\
Given a string `s` and a list of strings `word_dict`, return `True` if `s` can be **segmented** into a space-separated sequence of one or more dictionary words.

The same word in the dictionary may be reused multiple times.

**Example 1:**
```
Input:  s = "leetcode", word_dict = ["leet", "code"]
Output: True   # "leet" + "code" = "leetcode"
```

**Example 2:**
```
Input:  s = "applepenapple", word_dict = ["apple", "pen"]
Output: True   # "apple" + "pen" + "apple"
```

**Example 3:**
```
Input:  s = "catsandog", word_dict = ["cats", "dog", "sand", "and", "cat"]
Output: False
```

**Constraints:**
- 1 ≤ len(s) ≤ 300
- 1 ≤ len(word_dict) ≤ 1000
- All strings consist of lowercase English letters.
""",
        "python_tips": """\
**Key Python concept: DP with string slicing**

Use a boolean array `dp` where `dp[i]` means the first `i` characters of `s` can be segmented.

**The core idea:**
- `dp[0] = True` (empty string is always valid).
- For each position `i`, check every position `j < i`: if `dp[j]` is True and `s[j:i]` is in the dictionary, then `dp[i] = True`.
- Convert `word_dict` to a `set` for O(1) lookups.

**Useful Python syntax:**
```python3
word_set = set(word_dict)   # O(1) membership test
s[j:i]                       # substring from index j to i-1
```

**Time complexity:** O(n² * m) where n = len(s), m = average word length for slicing.
**Space complexity:** O(n).
""",
        "starter_code": """\
def word_break(s, word_dict):
    \"\"\"
    Args:
        s         (str):       the string to segment
        word_dict (list[str]): list of valid words

    Returns:
        bool: True if s can be segmented into dictionary words
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ("leetcode", ["leet", "code"]), "expected": True},
            {"input": ("applepenapple", ["apple", "pen"]), "expected": True},
            {"input": ("catsandog", ["cats", "dog", "sand", "and", "cat"]), "expected": False},
        ],
        "solution": """\
def word_break(s, word_dict):
    word_set = set(word_dict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[len(s)]
""",
    },

    {
        "id": 91,
        "title": "Maximum Product Subarray",
        "category": "Dynamic Programming",
        "difficulty": "Medium",
        "description": """\
Given an integer array `nums`, find a **contiguous subarray** that has the largest product, and return the product.

A subarray is a contiguous part of the array.

**Example 1:**
```
Input:  nums = [2, 3, -2, 4]
Output: 6   # subarray [2, 3] has the largest product
```

**Example 2:**
```
Input:  nums = [-2, 0, -1]
Output: 0   # subarray [0]
```

**Example 3:**
```
Input:  nums = [-2, 3, -4]
Output: 24  # the entire array: (-2) * 3 * (-4) = 24
```

**Constraints:**
- 1 ≤ len(nums) ≤ 2 * 10⁴
- -10 ≤ nums[i] ≤ 10
""",
        "python_tips": """\
**Key Python concept: Tracking both min and max**

Unlike maximum sum subarray, a negative number can flip the smallest product into the largest (and vice versa).

**The core idea:**
- Track `cur_max` and `cur_min` at each position.
- When you see a negative number, `cur_max` and `cur_min` swap roles.
- At each step: `cur_max = max(num, cur_max * num)` and `cur_min = min(num, cur_min * num)`.
- Keep a running `result = max(result, cur_max)`.

**Useful Python syntax:**
```python3
cur_max, cur_min = cur_min, cur_max   # simultaneous swap
max(a, b, c)                           # max of multiple values
```

**Time complexity:** O(n) — single pass.
**Space complexity:** O(1).
""",
        "starter_code": """\
def max_product(nums):
    \"\"\"
    Args:
        nums (list[int]): array of integers

    Returns:
        int: largest product of any contiguous subarray
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([2, 3, -2, 4],), "expected": 6},
            {"input": ([-2, 0, -1],), "expected": 0},
            {"input": ([-2, 3, -4],), "expected": 24},
        ],
        "solution": """\
def max_product(nums):
    result = nums[0]
    cur_max = cur_min = 1
    for num in nums:
        if num < 0:
            cur_max, cur_min = cur_min, cur_max
        cur_max = max(num, cur_max * num)
        cur_min = min(num, cur_min * num)
        result = max(result, cur_max)
    return result
""",
    },

    {
        "id": 92,
        "title": "Longest Increasing Subsequence",
        "category": "Dynamic Programming",
        "difficulty": "Medium",
        "description": """\
Given an integer array `nums`, return the **length of the longest strictly increasing subsequence**.

A subsequence is derived by deleting some (or no) elements without changing the order of the remaining elements.

**Example 1:**
```
Input:  nums = [10, 9, 2, 5, 3, 7, 101, 18]
Output: 4   # [2, 3, 7, 101] or [2, 5, 7, 101]
```

**Example 2:**
```
Input:  nums = [0, 1, 0, 3, 2, 3]
Output: 4   # [0, 1, 2, 3]
```

**Example 3:**
```
Input:  nums = [7, 7, 7, 7]
Output: 1   # all elements are the same
```

**Constraints:**
- 1 ≤ len(nums) ≤ 2500
- -10⁴ ≤ nums[i] ≤ 10⁴
""",
        "python_tips": """\
**Key Python concept: DP array where dp[i] = LIS ending at index i**

**The core idea:**
- Create `dp` where `dp[i]` = length of the longest increasing subsequence that **ends at** index `i`.
- Initialize every `dp[i] = 1` (each element alone is a subsequence of length 1).
- For each `i`, look at all `j < i`: if `nums[j] < nums[i]`, then `dp[i] = max(dp[i], dp[j] + 1)`.
- The answer is `max(dp)`.

**Useful Python syntax:**
```python3
dp = [1] * len(nums)   # all initialized to 1
max(dp)                  # find the global maximum
```

**Time complexity:** O(n²) — two nested loops.
**Space complexity:** O(n) — the dp array.
""",
        "starter_code": """\
def length_of_lis(nums):
    \"\"\"
    Args:
        nums (list[int]): array of integers

    Returns:
        int: length of the longest strictly increasing subsequence
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([10, 9, 2, 5, 3, 7, 101, 18],), "expected": 4},
            {"input": ([0, 1, 0, 3, 2, 3],), "expected": 4},
            {"input": ([7, 7, 7, 7],), "expected": 1},
        ],
        "solution": """\
def length_of_lis(nums):
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
""",
    },

    # ─────────────────────────────────────────────────────────────
    # 2D DYNAMIC PROGRAMMING
    # ─────────────────────────────────────────────────────────────
    {
        "id": 93,
        "title": "Unique Paths",
        "category": "Dynamic Programming",
        "difficulty": "Medium",
        "description": """\
A robot is located at the **top-left corner** of an `m x n` grid. The robot can only move **down** or **right** at any point.

The robot is trying to reach the **bottom-right corner** of the grid. How many possible **unique paths** are there?

**Example 1:**
```
Input:  m = 3, n = 7
Output: 28
```

**Example 2:**
```
Input:  m = 3, n = 2
Output: 3
Explanation: From top-left to bottom-right there are three paths:
  1. Right → Down → Down
  2. Down → Down → Right
  3. Down → Right → Down
```

**Constraints:**
- 1 ≤ m, n ≤ 100
""",
        "python_tips": """\
**Key Python concept: 2D DP table (or space-optimized 1D)**

**The core idea:**
- Create a 2D grid `dp[r][c]` = number of ways to reach cell (r, c).
- The first row and first column are all `1` (only one way: go straight right or straight down).
- For every other cell: `dp[r][c] = dp[r-1][c] + dp[r][c-1]` (come from above or from the left).

**Space optimization:** You only need the previous row, so you can use a 1D array:
```python3
dp = [1] * n
for _ in range(1, m):
    for j in range(1, n):
        dp[j] += dp[j - 1]
```

**Time complexity:** O(m * n).
**Space complexity:** O(n) with the optimized approach.
""",
        "starter_code": """\
def unique_paths(m, n):
    \"\"\"
    Args:
        m (int): number of rows
        n (int): number of columns

    Returns:
        int: number of unique paths from top-left to bottom-right
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": (3, 7), "expected": 28},
            {"input": (3, 2), "expected": 3},
            {"input": (1, 1), "expected": 1},
        ],
        "solution": """\
def unique_paths(m, n):
    dp = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    return dp[-1]
""",
    },

    {
        "id": 94,
        "title": "Longest Common Subsequence",
        "category": "Dynamic Programming",
        "difficulty": "Medium",
        "description": """\
Given two strings `text1` and `text2`, return the **length of their longest common subsequence**. If there is no common subsequence, return `0`.

A subsequence is a sequence that can be derived from the string by deleting some (or no) characters without changing the relative order of the remaining characters.

**Example 1:**
```
Input:  text1 = "abcde", text2 = "ace"
Output: 3   # "ace" is the longest common subsequence
```

**Example 2:**
```
Input:  text1 = "abc", text2 = "abc"
Output: 3   # "abc" — identical strings
```

**Example 3:**
```
Input:  text1 = "abc", text2 = "def"
Output: 0   # no common subsequence
```

**Constraints:**
- 1 ≤ len(text1), len(text2) ≤ 1000
- text1 and text2 consist of only lowercase English characters.
""",
        "python_tips": """\
**Key Python concept: Classic 2D DP**

**The core idea:**
- Build a 2D table `dp[i][j]` = length of LCS of `text1[:i]` and `text2[:j]`.
- If `text1[i-1] == text2[j-1]`: characters match, so `dp[i][j] = dp[i-1][j-1] + 1`.
- Otherwise: `dp[i][j] = max(dp[i-1][j], dp[i][j-1])` — skip one character from either string.
- Answer is `dp[len(text1)][len(text2)]`.

**Building the table:**
```python3
dp = [[0] * (n + 1) for _ in range(m + 1)]   # (m+1) x (n+1) grid of zeros
```

**Time complexity:** O(m * n).
**Space complexity:** O(m * n).
""",
        "starter_code": """\
def longest_common_subsequence(text1, text2):
    \"\"\"
    Args:
        text1 (str): first string
        text2 (str): second string

    Returns:
        int: length of the longest common subsequence
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ("abcde", "ace"), "expected": 3},
            {"input": ("abc", "abc"), "expected": 3},
            {"input": ("abc", "def"), "expected": 0},
        ],
        "solution": """\
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
""",
    },

    # ─────────────────────────────────────────────────────────────
    # GREEDY
    # ─────────────────────────────────────────────────────────────
    {
        "id": 95,
        "title": "Jump Game",
        "category": "Greedy",
        "difficulty": "Medium",
        "description": """\
You are given an integer array `nums`. You are initially positioned at the **first index**, and each element represents your **maximum jump length** at that position.

Return `True` if you can reach the **last index**, or `False` otherwise.

**Example 1:**
```
Input:  nums = [2, 3, 1, 1, 4]
Output: True
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
```

**Example 2:**
```
Input:  nums = [3, 2, 1, 0, 4]
Output: False
Explanation: You will always arrive at index 3, where the value is 0 — you are stuck.
```

**Constraints:**
- 1 ≤ len(nums) ≤ 10⁴
- 0 ≤ nums[i] ≤ 10⁵
""",
        "python_tips": """\
**Key Python concept: Greedy — track the farthest reachable index**

**The core idea:**
- Keep a variable `farthest` that tracks the maximum index you can reach.
- Walk through the array: at each index `i`, if `i > farthest`, you are stuck — return `False`.
- Otherwise update `farthest = max(farthest, i + nums[i])`.
- If you finish the loop, you can reach the end — return `True`.

**Why greedy works:**
You never need to go back. If a position is reachable, all positions before it are also reachable.

**Time complexity:** O(n) — single pass.
**Space complexity:** O(1).
""",
        "starter_code": """\
def can_jump(nums):
    \"\"\"
    Args:
        nums (list[int]): maximum jump length at each position

    Returns:
        bool: True if you can reach the last index
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([2, 3, 1, 1, 4],), "expected": True},
            {"input": ([3, 2, 1, 0, 4],), "expected": False},
            {"input": ([0],), "expected": True},
        ],
        "solution": """\
def can_jump(nums):
    farthest = 0
    for i in range(len(nums)):
        if i > farthest:
            return False
        farthest = max(farthest, i + nums[i])
    return True
""",
    },

    # ─────────────────────────────────────────────────────────────
    # INTERVALS
    # ─────────────────────────────────────────────────────────────
    {
        "id": 96,
        "title": "Insert Interval",
        "category": "Intervals",
        "difficulty": "Medium",
        "description": """\
You are given a list of **non-overlapping** intervals `intervals` where `intervals[i] = [start, end]`, sorted in ascending order by `start`. You are also given an interval `new_interval = [start, end]`.

**Insert** `new_interval` into `intervals` such that `intervals` is still sorted and non-overlapping (merge overlapping intervals if necessary).

Return `intervals` after the insertion.

**Example 1:**
```
Input:  intervals = [[1,3],[6,9]], new_interval = [2,5]
Output: [[1,5],[6,9]]
```

**Example 2:**
```
Input:  intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], new_interval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: [3,5], [6,7], [8,10] overlap with [4,8], so they merge into [3,10].
```

**Constraints:**
- 0 ≤ len(intervals) ≤ 10⁴
- intervals[i].length == 2
- Intervals are sorted and non-overlapping.
""",
        "python_tips": """\
**Key Python concept: Three-phase linear scan**

**The core idea:**
Split the problem into three parts:
1. **Before:** Add all intervals that end before the new interval starts (`interval[1] < new_interval[0]`).
2. **Overlap:** Merge all intervals that overlap with the new interval. Merge by taking `min` of starts and `max` of ends.
3. **After:** Add all remaining intervals.

**Useful Python syntax:**
```python3
result = []
result.append([start, end])  # add an interval
min(a, b)                     # merge start
max(a, b)                     # merge end
```

**Time complexity:** O(n) — single pass.
**Space complexity:** O(n) — the result list.
""",
        "starter_code": """\
def insert(intervals, new_interval):
    \"\"\"
    Args:
        intervals    (list[list[int]]): sorted non-overlapping intervals
        new_interval (list[int]):       interval to insert

    Returns:
        list[list[int]]: merged intervals after insertion
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([[1,3],[6,9]], [2,5]), "expected": [[1,5],[6,9]]},
            {"input": ([[1,2],[3,5],[6,7],[8,10],[12,16]], [4,8]), "expected": [[1,2],[3,10],[12,16]]},
            {"input": ([], [5,7]), "expected": [[5,7]]},
        ],
        "solution": """\
def insert(intervals, new_interval):
    result = []
    i = 0
    n = len(intervals)
    while i < n and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    result.append(new_interval)
    while i < n:
        result.append(intervals[i])
        i += 1
    return result
""",
    },

    {
        "id": 97,
        "title": "Merge Intervals",
        "category": "Intervals",
        "difficulty": "Medium",
        "description": """\
Given a list of `intervals` where `intervals[i] = [start, end]`, **merge all overlapping intervals** and return a list of the non-overlapping intervals that cover all the intervals in the input.

**Example 1:**
```
Input:  intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: [1,3] and [2,6] overlap, so they merge into [1,6].
```

**Example 2:**
```
Input:  intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: [1,4] and [4,5] are considered overlapping (they share endpoint 4).
```

**Constraints:**
- 1 ≤ len(intervals) ≤ 10⁴
- intervals[i].length == 2
- 0 ≤ start ≤ end ≤ 10⁴
""",
        "python_tips": """\
**Key Python concept: Sort then merge**

**The core idea:**
1. **Sort** intervals by their start time.
2. Walk through the sorted list. For each interval:
   - If the result list is empty or the current interval does **not** overlap with the last interval in the result, append it.
   - Otherwise, **merge** by extending the end of the last interval: `result[-1][1] = max(result[-1][1], interval[1])`.

**Two intervals overlap when:** the current start ≤ the previous end.

**Useful Python syntax:**
```python3
intervals.sort(key=lambda x: x[0])   # sort by start time
result[-1]                             # last element of the list
```

**Time complexity:** O(n log n) — dominated by sorting.
**Space complexity:** O(n) — the result list.
""",
        "starter_code": """\
def merge(intervals):
    \"\"\"
    Args:
        intervals (list[list[int]]): list of intervals [start, end]

    Returns:
        list[list[int]]: merged non-overlapping intervals
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([[1,3],[2,6],[8,10],[15,18]],), "expected": [[1,6],[8,10],[15,18]]},
            {"input": ([[1,4],[4,5]],), "expected": [[1,5]]},
            {"input": ([[1,4],[0,4]],), "expected": [[0,4]]},
        ],
        "solution": """\
def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    result = []
    for interval in intervals:
        if not result or result[-1][1] < interval[0]:
            result.append(interval)
        else:
            result[-1][1] = max(result[-1][1], interval[1])
    return result
""",
    },

    {
        "id": 98,
        "title": "Non-overlapping Intervals",
        "category": "Intervals",
        "difficulty": "Medium",
        "description": """\
Given an array of intervals `intervals` where `intervals[i] = [start_i, end_i]`, return the **minimum number of intervals you need to remove** to make the rest of the intervals non-overlapping.

Two intervals `[a, b]` and `[c, d]` are **non-overlapping** if `b <= c` or `d <= a`.

**Example 1:**
```
Input:  intervals = [[1,2],[2,3],[3,4],[1,3]]
Output: 1
Explanation: Remove [1,3] and the rest are non-overlapping.
```

**Example 2:**
```
Input:  intervals = [[1,2],[1,2],[1,2]]
Output: 2
Explanation: You need to remove two [1,2] to make the rest non-overlapping.
```

**Example 3:**
```
Input:  intervals = [[1,2],[2,3]]
Output: 0
Explanation: Already non-overlapping.
```

**Constraints:**
- 1 <= len(intervals) <= 10^5
- intervals[i].length == 2
- -5 * 10^4 <= start_i < end_i <= 5 * 10^4
""",
        "python_tips": """\
**Key Python concept: Greedy algorithm with sorting**

The trick is to sort intervals by **end time** and greedily keep as many as possible.

**The core idea:**
- Sort by end time so you always pick the interval that finishes earliest.
- Walk through: if the current interval starts **before** the previous one ends, it overlaps — remove it (increment counter).
- If it doesn't overlap, update your "last kept end" to this interval's end.

**Why sort by end time?** Finishing early leaves the most room for future intervals.

```python3
intervals.sort(key=lambda x: x[1])  # sort by end time
```

**Time complexity:** O(n log n) for sorting.
**Space complexity:** O(1) extra space.
""",
        "starter_code": """\
def erase_overlap_intervals(intervals):
    \"\"\"
    Args:
        intervals (list[list[int]]): list of [start, end] intervals

    Returns:
        int: minimum number of intervals to remove
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([[1,2],[2,3],[3,4],[1,3]],), "expected": 1},
            {"input": ([[1,2],[1,2],[1,2]],), "expected": 2},
            {"input": ([[1,2],[2,3]],), "expected": 0},
        ],
        "solution": """\
def erase_overlap_intervals(intervals):
    intervals.sort(key=lambda x: x[1])
    count = 0
    end = float('-inf')
    for start_i, end_i in intervals:
        if start_i < end:
            count += 1
        else:
            end = end_i
    return count
""",
    },

    {
        "id": 99,
        "title": "Meeting Rooms",
        "category": "Intervals",
        "difficulty": "Easy",
        "description": """\
Given an array of meeting time intervals `intervals` where `intervals[i] = [start_i, end_i]`, determine if a person could **attend all meetings**.

A person cannot attend two meetings that overlap.

**Example 1:**
```
Input:  intervals = [[0,30],[5,10],[15,20]]
Output: False
Explanation: [0,30] and [5,10] overlap.
```

**Example 2:**
```
Input:  intervals = [[7,10],[2,4]]
Output: True
Explanation: No meetings overlap.
```

**Example 3:**
```
Input:  intervals = []
Output: True
Explanation: No meetings at all — no conflicts.
```

**Constraints:**
- 0 <= len(intervals) <= 10^4
- intervals[i].length == 2
- 0 <= start_i < end_i <= 10^6
""",
        "python_tips": """\
**Key Python concept: Sorting and pairwise comparison**

If meetings are sorted by start time, you only need to check if each meeting starts before the previous one ends.

**The core idea:**
- Sort intervals by start time.
- Walk through pairs of consecutive meetings.
- If `intervals[i][0] < intervals[i-1][1]`, there's an overlap — return False.
- If you make it through all pairs, return True.

```python3
intervals.sort()  # sorts by first element by default
```

**Time complexity:** O(n log n) for sorting.
**Space complexity:** O(1) extra space.
""",
        "starter_code": """\
def can_attend_meetings(intervals):
    \"\"\"
    Args:
        intervals (list[list[int]]): list of [start, end] meeting times

    Returns:
        bool: True if person can attend all meetings
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([[0,30],[5,10],[15,20]],), "expected": False},
            {"input": ([[7,10],[2,4]],), "expected": True},
            {"input": ([],), "expected": True},
        ],
        "solution": """\
def can_attend_meetings(intervals):
    intervals.sort()
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i-1][1]:
            return False
    return True
""",
    },

    {
        "id": 100,
        "title": "Meeting Rooms II",
        "category": "Intervals",
        "difficulty": "Medium",
        "description": """\
Given an array of meeting time intervals `intervals` where `intervals[i] = [start_i, end_i]`, return the **minimum number of conference rooms** required.

**Example 1:**
```
Input:  intervals = [[0,30],[5,10],[15,20]]
Output: 2
Explanation: [0,30] overlaps with [5,10], so you need 2 rooms.
             [15,20] can reuse the room freed by [5,10].
```

**Example 2:**
```
Input:  intervals = [[7,10],[2,4]]
Output: 1
Explanation: The meetings don't overlap — 1 room is enough.
```

**Example 3:**
```
Input:  intervals = [[1,5],[2,6],[3,7]]
Output: 3
Explanation: All three meetings overlap at time 3.
```

**Constraints:**
- 1 <= len(intervals) <= 10^4
- 0 <= start_i < end_i <= 10^6
""",
        "python_tips": """\
**Key Python concept: Min-heap for tracking room availability**

Use a min-heap to track when each room becomes free.

**The core idea:**
- Sort meetings by start time.
- Use a heap where each element is the end time of a meeting in a room.
- For each meeting: if the earliest room frees up before this meeting starts, reuse it (pop from heap).
- Push the current meeting's end time onto the heap.
- The heap size at the end = number of rooms needed.

```python3
import heapq
heapq.heappush(heap, end_time)   # add a room
heapq.heappop(heap)              # free earliest room
heap[0]                          # peek at earliest end time
```

**Time complexity:** O(n log n).
**Space complexity:** O(n).
""",
        "starter_code": """\
def min_meeting_rooms(intervals):
    \"\"\"
    Args:
        intervals (list[list[int]]): list of [start, end] meeting times

    Returns:
        int: minimum number of conference rooms required
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([[0,30],[5,10],[15,20]],), "expected": 2},
            {"input": ([[7,10],[2,4]],), "expected": 1},
            {"input": ([[1,5],[2,6],[3,7]],), "expected": 3},
        ],
        "solution": """\
def min_meeting_rooms(intervals):
    import heapq
    intervals.sort()
    heap = []
    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heappop(heap)
        heapq.heappush(heap, end)
    return len(heap)
""",
    },

    # ─────────────────────────────────────────────────────────────
    # MATH & GEOMETRY
    # ─────────────────────────────────────────────────────────────
    {
        "id": 101,
        "title": "Rotate Image",
        "category": "Math & Geometry",
        "difficulty": "Medium",
        "description": """\
You are given an `n x n` 2D matrix representing an image. Rotate the image by **90 degrees clockwise** **in place**.

After rotating, return the matrix.

**Example 1:**
```
Input:  matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]
```

**Example 2:**
```
Input:  matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
```

**Constraints:**
- n == matrix.length == matrix[i].length
- 1 <= n <= 20
- -1000 <= matrix[i][j] <= 1000
""",
        "python_tips": """\
**Key Python concept: Matrix transposition and reversal**

A 90-degree clockwise rotation can be done in two simple steps:
1. **Transpose** the matrix (swap rows and columns: `matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]`)
2. **Reverse each row** (`row.reverse()`)

**Why this works:**
- Transposing flips the matrix along its diagonal.
- Reversing each row then gives the clockwise rotation.

```python3
for i in range(n):
    for j in range(i + 1, n):
        matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
```

**Time complexity:** O(n²).
**Space complexity:** O(1) — done in place.
""",
        "starter_code": """\
def rotate(matrix):
    \"\"\"
    Args:
        matrix (list[list[int]]): n x n 2D matrix

    Returns:
        list[list[int]]: the matrix after 90-degree clockwise rotation
    \"\"\"
    # your code here (modify matrix in place, then return it)
    pass
""",
        "test_cases": [
            {"input": ([[1,2,3],[4,5,6],[7,8,9]],), "expected": [[7,4,1],[8,5,2],[9,6,3]]},
            {"input": ([[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]],), "expected": [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]},
            {"input": ([[1]],), "expected": [[1]]},
        ],
        "solution": """\
def rotate(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()
    return matrix
""",
    },

    {
        "id": 102,
        "title": "Spiral Matrix",
        "category": "Math & Geometry",
        "difficulty": "Medium",
        "description": """\
Given an `m x n` matrix, return all elements of the matrix in **spiral order**.

Spiral order means: traverse right across the top row, then down the right column, then left across the bottom row, then up the left column, and repeat inward.

**Example 1:**
```
Input:  matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]
```

**Example 2:**
```
Input:  matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]
```

**Constraints:**
- m == matrix.length
- n == matrix[i].length
- 1 <= m, n <= 10
- -100 <= matrix[i][j] <= 100
""",
        "python_tips": """\
**Key Python concept: Boundary tracking with four pointers**

Maintain four boundaries — `top`, `bottom`, `left`, `right` — and shrink them after each traversal.

**The core idea:**
1. Go **right** along `top` row, then `top += 1`
2. Go **down** along `right` column, then `right -= 1`
3. Go **left** along `bottom` row (if `top <= bottom`), then `bottom -= 1`
4. Go **up** along `left` column (if `left <= right`), then `left += 1`
5. Repeat while `top <= bottom` and `left <= right`

**Time complexity:** O(m * n) — visit every element once.
**Space complexity:** O(1) extra (not counting output).
""",
        "starter_code": """\
def spiral_order(matrix):
    \"\"\"
    Args:
        matrix (list[list[int]]): m x n matrix

    Returns:
        list[int]: elements in spiral order
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([[1,2,3],[4,5,6],[7,8,9]],), "expected": [1,2,3,6,9,8,7,4,5]},
            {"input": ([[1,2,3,4],[5,6,7,8],[9,10,11,12]],), "expected": [1,2,3,4,8,12,11,10,9,5,6,7]},
            {"input": ([[1]],), "expected": [1]},
        ],
        "solution": """\
def spiral_order(matrix):
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        top += 1
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1
        if top <= bottom:
            for j in range(right, left - 1, -1):
                result.append(matrix[bottom][j])
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1
    return result
""",
    },

    {
        "id": 103,
        "title": "Set Matrix Zeroes",
        "category": "Math & Geometry",
        "difficulty": "Medium",
        "description": """\
Given an `m x n` integer matrix, if an element is `0`, set its **entire row and column** to `0`. You must do it **in place**.

After modifying, return the matrix.

**Example 1:**
```
Input:  matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]
```

**Example 2:**
```
Input:  matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
```

**Constraints:**
- m == matrix.length
- n == matrix[0].length
- 1 <= m, n <= 200
- -2^31 <= matrix[i][j] <= 2^31 - 1
""",
        "python_tips": """\
**Key Python concept: Using sets to track zero positions**

**The core idea (O(m+n) space approach):**
1. First pass: find all rows and columns that contain a zero.
2. Second pass: set every cell to zero if its row or column is in the zero set.

```python3
zero_rows = set()
zero_cols = set()
```

**Time complexity:** O(m * n).
**Space complexity:** O(m + n).
""",
        "starter_code": """\
def set_zeroes(matrix):
    \"\"\"
    Args:
        matrix (list[list[int]]): m x n matrix

    Returns:
        list[list[int]]: the matrix after setting zeroes
    \"\"\"
    # your code here (modify matrix in place, then return it)
    pass
""",
        "test_cases": [
            {"input": ([[1,1,1],[1,0,1],[1,1,1]],), "expected": [[1,0,1],[0,0,0],[1,0,1]]},
            {"input": ([[0,1,2,0],[3,4,5,2],[1,3,1,5]],), "expected": [[0,0,0,0],[0,4,5,0],[0,3,1,0]]},
        ],
        "solution": """\
def set_zeroes(matrix):
    m, n = len(matrix), len(matrix[0])
    zero_rows = set()
    zero_cols = set()
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 0:
                zero_rows.add(i)
                zero_cols.add(j)
    for i in range(m):
        for j in range(n):
            if i in zero_rows or j in zero_cols:
                matrix[i][j] = 0
    return matrix
""",
    },

    # ─────────────────────────────────────────────────────────────
    # BIT MANIPULATION
    # ─────────────────────────────────────────────────────────────
    {
        "id": 104,
        "title": "Number of 1 Bits",
        "category": "Bit Manipulation",
        "difficulty": "Easy",
        "description": """\
Given a positive integer `n`, return the number of **set bits** (1s) in its binary representation. This is also known as the **Hamming weight**.

**Example 1:**
```
Input:  n = 11    (binary: 1011)
Output: 3
```

**Example 2:**
```
Input:  n = 128   (binary: 10000000)
Output: 1
```

**Example 3:**
```
Input:  n = 255   (binary: 11111111)
Output: 8
```

**Constraints:**
- 0 <= n <= 2^31 - 1
""",
        "python_tips": """\
**Key Python concept: Bitwise AND and right shift**

You can check and strip bits one at a time using bitwise operators.

**Approach 1 — Check each bit:**
- `n & 1` gives you the last bit (0 or 1).
- `n >>= 1` shifts all bits right by 1 (drops the last bit).
- Repeat until `n == 0`.

**Approach 2 — Brian Kernighan's trick:**
- `n & (n - 1)` clears the lowest set bit.
- Count how many times you can do this before `n == 0`.

```python3
n & 1       # last bit: 0 or 1
n >> 1      # right shift by 1
n & (n - 1) # clear lowest set bit
```

**Time complexity:** O(number of bits) = O(32) = O(1).
""",
        "starter_code": """\
def hamming_weight(n):
    \"\"\"
    Args:
        n (int): a non-negative integer

    Returns:
        int: number of set bits (1s) in binary representation
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": (11,), "expected": 3},
            {"input": (128,), "expected": 1},
            {"input": (255,), "expected": 8},
        ],
        "solution": """\
def hamming_weight(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count
""",
    },

    {
        "id": 105,
        "title": "Counting Bits",
        "category": "Bit Manipulation",
        "difficulty": "Easy",
        "description": """\
Given an integer `n`, return an array `ans` of length `n + 1` such that for each `i` (0 <= i <= n), `ans[i]` is the **number of 1s** in the binary representation of `i`.

**Example 1:**
```
Input:  n = 2
Output: [0, 1, 1]
Explanation: 0 -> 0, 1 -> 1, 2 -> 10
```

**Example 2:**
```
Input:  n = 5
Output: [0, 1, 1, 2, 1, 2]
Explanation: 0->0, 1->1, 2->10, 3->11, 4->100, 5->101
```

**Constraints:**
- 0 <= n <= 10^5
""",
        "python_tips": """\
**Key Python concept: Dynamic programming with bit tricks**

You can build on previously computed answers instead of counting bits from scratch each time.

**The core insight:**
- `i >> 1` is `i` with the last bit removed — you already know its count!
- `i & 1` tells you if the last bit is 1.
- So: `ans[i] = ans[i >> 1] + (i & 1)`

**Example:** `ans[5] = ans[5 >> 1] + (5 & 1) = ans[2] + 1 = 1 + 1 = 2`

```python3
ans = [0] * (n + 1)
for i in range(1, n + 1):
    ans[i] = ans[i >> 1] + (i & 1)
```

**Time complexity:** O(n).
**Space complexity:** O(n) for the output array.
""",
        "starter_code": """\
def count_bits(n):
    \"\"\"
    Args:
        n (int): a non-negative integer

    Returns:
        list[int]: array where ans[i] = number of 1s in binary of i
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": (2,), "expected": [0, 1, 1]},
            {"input": (5,), "expected": [0, 1, 1, 2, 1, 2]},
            {"input": (0,), "expected": [0]},
        ],
        "solution": """\
def count_bits(n):
    ans = [0] * (n + 1)
    for i in range(1, n + 1):
        ans[i] = ans[i >> 1] + (i & 1)
    return ans
""",
    },

    {
        "id": 106,
        "title": "Reverse Bits",
        "category": "Bit Manipulation",
        "difficulty": "Easy",
        "description": """\
Reverse the bits of a given **32-bit unsigned integer**.

**Example 1:**
```
Input:  n = 43261596   (binary: 00000010100101000001111010011100)
Output: 964176192      (binary: 00111001011110000010100101000000)
```

**Example 2:**
```
Input:  n = 4294967293  (binary: 11111111111111111111111111111101)
Output: 3221225471      (binary: 10111111111111111111111111111111)
```

**Constraints:**
- The input is a 32-bit unsigned integer.
""",
        "python_tips": """\
**Key Python concept: Bit-by-bit reversal**

Extract each bit from the right and build the reversed number from the left.

**The core idea:**
- Loop 32 times (one for each bit).
- Each iteration: shift `result` left by 1, then add the last bit of `n` (`n & 1`).
- Shift `n` right by 1 to process the next bit.

```python3
result = 0
for i in range(32):
    result = (result << 1) | (n & 1)
    n >>= 1
```

**Time complexity:** O(32) = O(1).
**Space complexity:** O(1).
""",
        "starter_code": """\
def reverse_bits(n):
    \"\"\"
    Args:
        n (int): a 32-bit unsigned integer

    Returns:
        int: the integer with reversed bits
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": (43261596,), "expected": 964176192},
            {"input": (4294967293,), "expected": 3221225471},
            {"input": (0,), "expected": 0},
        ],
        "solution": """\
def reverse_bits(n):
    result = 0
    for i in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result
""",
    },

    {
        "id": 107,
        "title": "Missing Number",
        "category": "Bit Manipulation",
        "difficulty": "Easy",
        "description": """\
Given an array `nums` containing `n` distinct numbers in the range `[0, n]`, return the **only number in the range that is missing** from the array.

**Example 1:**
```
Input:  nums = [3, 0, 1]
Output: 2
Explanation: n = 3 since there are 3 numbers. 2 is missing.
```

**Example 2:**
```
Input:  nums = [0, 1]
Output: 2
Explanation: n = 2 since there are 2 numbers. 2 is missing.
```

**Example 3:**
```
Input:  nums = [9,6,4,2,3,5,7,0,1]
Output: 8
```

**Constraints:**
- n == nums.length
- 0 <= nums[i] <= n
- All numbers in nums are **unique**.
""",
        "python_tips": """\
**Key Python concept: XOR or math formula**

**Approach 1 — Math (Gauss's formula):**
- The sum of `0` to `n` is `n * (n + 1) // 2`.
- Subtract the actual sum of the array to find the missing number.

```python3
n = len(nums)
expected_sum = n * (n + 1) // 2
return expected_sum - sum(nums)
```

**Approach 2 — XOR:**
- XOR all numbers `0..n` with all numbers in the array.
- Every number that appears twice cancels out (`x ^ x = 0`).
- The only number left is the missing one.

**Time complexity:** O(n).
**Space complexity:** O(1).
""",
        "starter_code": """\
def missing_number(nums):
    \"\"\"
    Args:
        nums (list[int]): array of n distinct numbers from range [0, n]

    Returns:
        int: the missing number
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": ([3, 0, 1],), "expected": 2},
            {"input": ([0, 1],), "expected": 2},
            {"input": ([9,6,4,2,3,5,7,0,1],), "expected": 8},
        ],
        "solution": """\
def missing_number(nums):
    n = len(nums)
    return n * (n + 1) // 2 - sum(nums)
""",
    },

    {
        "id": 108,
        "title": "Sum of Two Integers",
        "category": "Bit Manipulation",
        "difficulty": "Medium",
        "description": """\
Given two integers `a` and `b`, return the **sum of the two integers** without using the `+` or `-` operators.

**Example 1:**
```
Input:  a = 1, b = 2
Output: 3
```

**Example 2:**
```
Input:  a = 2, b = 3
Output: 5
```

**Example 3:**
```
Input:  a = -1, b = 1
Output: 0
```

**Constraints:**
- -1000 <= a, b <= 1000
""",
        "python_tips": """\
**Key Python concept: Bitwise addition with XOR and AND**

In binary, addition works like this:
- **XOR (`^`)** gives the sum without carries (like adding each bit ignoring overflow).
- **AND (`&`) then left shift (`<< 1`)** gives the carry bits.
- Repeat until there are no more carries.

**The catch in Python:** Python integers have unlimited size, so negative numbers don't naturally wrap around like in C/Java. You need a **32-bit mask**.

```python3
MASK = 0xFFFFFFFF      # 32-bit mask (all 1s)
MAX  = 0x7FFFFFFF      # max positive 32-bit int

a & MASK               # keep only lower 32 bits
```

**The algorithm:**
1. While carry is not zero: compute new sum (XOR), new carry (AND << 1), mask both.
2. If result > MAX, it's negative in 32-bit — convert with `~(result ^ MASK)`.

**Time complexity:** O(32) = O(1).
""",
        "starter_code": """\
def get_sum(a, b):
    \"\"\"
    Args:
        a (int): first integer
        b (int): second integer

    Returns:
        int: sum of a and b (without using + or -)
    \"\"\"
    # your code here
    pass
""",
        "test_cases": [
            {"input": (1, 2), "expected": 3},
            {"input": (2, 3), "expected": 5},
            {"input": (-1, 1), "expected": 0},
        ],
        "solution": """\
def get_sum(a, b):
    MASK = 0xFFFFFFFF
    MAX = 0x7FFFFFFF
    a &= MASK
    b &= MASK
    while b:
        carry = (a & b) << 1
        a = a ^ b
        b = carry & MASK
        a &= MASK
    return a if a <= MAX else ~(a ^ MASK)
""",
    },
]

# Convenience helpers
CATEGORIES = sorted(set(p["category"] for p in PROBLEMS))
DIFFICULTY_ORDER = {"Easy": 0, "Medium": 1, "Hard": 2}

# ─────────────────────────────────────────────────────────────────────────────
# Original Blind 75 problem IDs
# These are the 75 problems from the famous Blind 75 interview prep list.
# The other 33 problems in this repo are bonus extras.
# ─────────────────────────────────────────────────────────────────────────────
BLIND75_IDS: set[int] = {
    # Arrays & Hashing (10)
    1, 2, 3, 4, 5, 7, 16, 30, 31, 91,
    # String / Sliding Window (7)
    6, 8, 11, 12, 14, 18, 19,
    # Stack (1)
    21,
    # Linked List (6)
    34, 35, 36, 37, 40, 43,
    # Trees (14)
    10, 45, 48, 49, 50, 51, 54, 55, 56, 57, 58, 59, 60, 61,
    # Heap (2)
    13, 67,
    # Backtracking (2)
    69, 73,
    # Graphs (8)
    15, 76, 77, 78, 79, 80, 81, 82,
    # Dynamic Programming (12)
    83, 84, 85, 86, 87, 88, 89, 90, 92, 93, 94, 95,
    # Intervals (5)
    96, 97, 98, 99, 100,
    # Matrix (3)
    101, 102, 103,
    # Bit Manipulation (5)
    104, 105, 106, 107, 108,
}


def get_problems_by_category(category: str):
    return [p for p in PROBLEMS if p["category"] == category]


def get_problem_by_id(problem_id: int):
    for p in PROBLEMS:
        if p["id"] == problem_id:
            return p
    return None
