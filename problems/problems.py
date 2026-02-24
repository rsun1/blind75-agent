# All 75 Blind LeetCode problems
# Full content for the first 10 core problems; concise stubs with test cases for the rest.

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
```python
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
```python
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

```python
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

```python
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

```python
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
```python
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
```python
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

```python
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

```python
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

```python
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
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val   = val
        self.left  = left
        self.right = right
```

To invert a tree, simply swap the left and right children — then recursively do the same for both subtrees.

**Recursive approach:**
```python
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
        "test_cases": [],
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
]

# Convenience helpers
CATEGORIES = sorted(set(p["category"] for p in PROBLEMS))
DIFFICULTY_ORDER = {"Easy": 0, "Medium": 1, "Hard": 2}


def get_problems_by_category(category: str):
    return [p for p in PROBLEMS if p["category"] == category]


def get_problem_by_id(problem_id: int):
    for p in PROBLEMS:
        if p["id"] == problem_id:
            return p
    return None
