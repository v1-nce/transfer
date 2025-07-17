# min(i, n-1-i) creates the pattern: 1234321 sequentially.
# max(dict.items(), lambda x: x[1])[0] returns maximum value in dict
# lst.sort(key = lambda x: (x[1], x[2])) sort on many variables
# len(set(x)) == 1: checks for uniqueness
# while current != None: current = current.after()
# return function(x) or function(y) recursion checks through the tree
# [set() for _ in range(9)] creates a list of sets
# Window sum = + new value - old value in a queue block
# .split, .join(str), .rsplit,
# lst.insert(value, index)
# Set has add update(iterable) remove clear pop .union() .intersection() .difference(), .issubset, .issuperset.
#
# def add(a, b, c):
#     return a + b + c
# numbers = [1, 2, 3]
# print(*map(lambda x: x, numbers))
# result = add(*map(lambda x: x, numbers))
# print(result)  # Output: 6
#
# ValueError ZeroDivisionError TypeError IndexError AttributeError
# Selection Sort: Unstable, in-place
# Merge Sort: Stable, not in-place
# Bubble Sort: Stable, in-place
# Insertion Sort: Stable, in-place
# Quick Sort: Unstable
# **kwargs collects input as a dictionary.
# dict[key] = dict.get(key, 0) + val


# Test Prime
def test_prime(n):
    if n == 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

# Fibonacci
def make_fibonacci(n):
    if n == 0:
        return (0,)
    fib = (0,1)
    for x in range(n-1):
        fib += (fib[-1] + fib[-2],)
    return fib

# Ruler Pyramid
def ruler(n):
    if n == 1:
        return "1"
    prev = ruler(n-1)
    return prev + str(n) + prev

# Dynamic Tower of Hanoi
def hanoi(n, source, destination, auxiliary):
    if n == 1:
        return ((source, destination),)
    else:
        move = hanoi(n-1, source, auxiliary, destination)
        move += ((source, destination),)
        move += hanoi(n-1, auxiliary, destination, source)
        return move
def move_tower(size, A, C, B):
    if size == 0:
        return True
    else:
        move_tower(size-1, A, B, C)
        print("move top disk from", A, "to", C)
        move_tower(size-1, B, C, A) # move the bottom disk then the rest of the disks?

# Counting Change
def cc_listing(a, n, partial_ans):
    if a == 0:
        print(partial_ans[:-1])
        return 1
    elif a < 0:
        return 0
    elif n == 0:
        return 0
    else:
        return cc_listing(a-denomination(n), n, partial_ans + str(denomination(n)) + "+") + cc_listing(a, n-1, partial_ans)
denom = (1,5,10,20,50)
def denomination(n):
    return denom[n-1]

# Counting Change Memoized
def count_change(amount, coins):
    memo = {}
    def helper(a, i):
        if a == 0:
            return 1
        if a < 0 or i == len(coins):
            return 0
        if (a, i) in memo:
            return memo[(a, i)]
        memo[(a, i)] = helper(a - coins[i], i) + helper(a, i + 1)
        return memo[(a, i)]
    return helper(amount, 0)

# Counting Change Dynamic Programming?
kinds_of_coins = [1,5,10,20,50,100]
def count_ways(amount, kinds_of_coins):
    dp = [0] * (amount + 1)
    dp[0] = 1 
    for coin in kinds_of_coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    return dp[amount]
def dp_cc(a, d):
    table = []
    oneline = [0] * (d + 1)
    for i in range(a + 1):
        table.append(list(oneline))
    for i in range(1, d+1):   # Row>amount, Column>first col coins
        table[0][i] = 1
    for col in range(1, d + 1):
        for row in range(1, a + 1):
            if row - coins[col-1] < 0: # Base case 
                table[row][col] = table[row][col-1]
            else:
                # Use current coin + Amount of ways to make without using that coin
                table[row][col] = table[row - coins[col - 1]][col] + table[row][col-1]
    print(table)
    return table

# Accumulate
def accumulate(fn, initial, seqence):
    if seq == ():
        return initial
    else:
        return fn(seq[0], accumulate(fn, initial, sequence[1:]))

# Count Leaves
def count_leaves(tree):
    if tree == ():
        return 0
    elif type(tree) != tuple:
        return 1
    else:
        return count_leaves(tree[0]) + count_leaves(tree[1:])

# Count Nodes
def sum_nodes(tree):
    if tree == ():
        return 0
    elif not isinstance(tree, tuple):
        return tree
    else:
        node, left, right = tree
        return node + sum_nodes(left) + sum_nodes(right)

# Tree Flatten
def tree_flatten(tree):
    if tree == ():
        return ()
    elif type(tree) != tuple:
        return (tree,)
    else:
        return tree_flatten(tree[0]) + tree_flatten(tree[1:])
def flatten_optimised(tree):
    def helper(tree, acc):
        if is_empty_tree(tree):
            return acc
        helper(left_branch(tree), acc)
        acc.append(entry(tree))
        helper(right_branch(tree), acc)
        return acc
    return helper(tree, [])

# Tree Scale
def tree_scale(tree, factor):
    def scale_fn(subtree):
        if type(subtree) != tuple:
            return factor * subtree
        else:
            tree_scale(subtree, factor)
    return tuple(map(scale_fn, tree))

# Tree Copy
def tree_copy(tree):
    def copier(subtree):
        if type(subtree) != tuple:
            return subtree
        else:
            return copy_tree(subtree)
    return tuple(map(copier, tree))

# Tree Filter
def tree_filter(tree, predicate):
    def filter_fn(subtree):
        if not isinstance(subtree, tuple):
            return subtree if predicate(subtree) else None
        else:
            filtered_children = tuple(filter(None, map(filter_fn, subtree)))
            return filtered_children if filtered_children else None
    return filter_fn(tree)
def tree_filter(tree, predicate):
    def filter_fn(subtree):
        if not isinstance(subtree, tuple):
            return subtree if predicate(subtree) else None
        else:
            filtered_children = []
            for child in subtree:
                filtered_child = filter_fn(child)
                if filtered_child is not None:
                    filtered_children.append(filtered_child)
            return tuple(filtered_children) if filtered_children else None
    return filter_fn(tree)

# Create Balanced/Binary Tree
def balanced_tree(sorted_list):
    if not sorted_list:
        return 
    mid = len(sorted_list) // 2
    root = sorted_list[mid]
    left = balanced_tree(sorted_list[:mid])
    right = balanced_tree(sorted_list[mid+1:])
    return [left, root, right]

# Create Binary Tree from list (First element will be the root)
def insert_bst(tree, value):
    if tree is None:
        return [value, None, None]
    if value < tree[0]:
        tree[1] = insert_bst(tree[1], value)
    else:
        tree[2] = insert_bst(tree[2], value)
    return tree
def build_bst_from_unsorted(lst):
    tree = None
    for value in lst:
        tree = insert_bst(tree, value)
    return tree

# Operate on Nested Lists
def apply_to_nested(data, func):
    result = []
    for item in data:
        if isinstance(item, list):
            nested_result = apply_to_nested(item, func)
            result.append(nested_result)
        else:
            transformed_item = func(item)
            result.append(transformed_item)
    return result

# Multi-Conditional Sorting
def sort_grade_then_name(x):
    length = len(x)
    i = 0
    for i in range(length):
        smallest = i
        for j in range(i+1, length):
            if x[smallest][1] > x[j][1]:
                smallest = j
            elif x[smallest][1] == x[j][1]:
                if x[smallest][0] > x[j][0]:
                    smallest = j
        x[i], x[smallest] = x[smallest], x[i]
    return x

# Another Way to Build a Binary Tree
def binary_tree(iTuple):
    root = find_root(iTuple)
    def build(node):
        if node == -1:
            return ()
        left_child = -1
        right_child = -1
        for a, b, c in iTuple: # finds the exact tuple node in iTuple
            if a == node:
                left_child = b
                right_child = c
                break
        left_subtree = build(left_child)
        right_subtree = build(right_child)
        return (node, left_subtree, right_subtree)
    return build(root)

###### You are actually only caring about the root and nothing else
###### Left and right subtree is only a "medium" to get the next root.

# Binary Search
def binary_search(key, seq):
    def helper(low, high):
        if low > high:
            return False
        mid = (low + high) // 2
        if key == seq[mid]:
            return True
        elif key < seq[mid]:
            return helper(0, mid - 1)
        else:
            return helper(mid + 1, high)
    return helper(0, len(seq) - 1)

# Selection Sort (Not Inplace but stable)
def selection(lst):
   sort=[]
   while lst:
       smallest = lst[0]
       for element in lst:
           if element < smallest:
               smallest = element
       lst.remove(smallest)
       sort.append(smallest)
   # lst = sort.copy() - does not work as needed
   lst.extend(sort)
 
# Selection Sort (Inplace but not Stable)
def selection(lst):
   for i in range(len(lst) - 1):
       for j in range(i+1, len(lst)):
           if lst[i] > lst[j]:
               lst[i], lst[j] = lst[j], lst[i]

# Merge Sort
def merge(left, right):
    result = []
    while left and right:
        if left[0] < right[0]:
            result.append(left[0])
            left.remove(left[0])
        else:
            result.append(right[0])
            right.remove(right[0])
    results.extend(left)
    result.extend(right)
    return result
def merge_sort(lst):
    if len(lst) < 2:
        return lst
    left = merge_sort(lst[:len(lst)//2])
    right = merge_sort(lst[len(lst)//2:])
    return merge(left, right)

# Bubble Sort (Optimised)
def bubble(x):
    length = len(x)
    is_sorted = False
    while not is_sorted:
        is_sorted = True  
        for i in range(length - 1):
            if x[i] > x[i + 1]:
                x[i], x[i + 1] = x[i + 1], x[i]
                is_sorted = False  
    return x
def bubble(lst):
    last_change = len(lst) - 1
    while last_change != 0:
        new_j = last_change
        for i in range(new_j):
            if lst[i] > lst[i + 1]:
                lst[i], lst[i + 1] = lst[i + 1], lst[i]
                last_change = i

# Insertion Sort
def insertion_sort(lst):
    indexing_length = range(1, len(lst))
    for i in indexing_length:
        value_to_sort = lst[i]
        while lst[i-1] > value_to_sort and i>0:
            lst[i], lst[i-1] = lst[i-1], lst[i]
            i = i-1
    return lst

# Quicksort
def quicksort(lst):
    if len(lst) <= 1:
        return lst
    pivot = lst[-1]
    left = [x for x in lst[:-1] if x <= pivot]
    middle = [x for x in lst if x == pivot] # Those with the same value
    right = [x for x in lst[:-1] if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# Quick Select (Workings)
def parition(lst, left, right):
    pivot_index = random.randint(left, right)
    lst[pivot_index], lst[right] = lst[right], lst[pivot_index]
    x = lst[right]
    i = left
    for j in range(left, right):
        if lst[j] <= x:
            lst[i], lst[j] = lst[j], lst[i]
            i += 1
    lst[i], lst[right] = lst[right], lst[i]
    return i
def quick_select(lst, left, right, k): # kth smallest element
    if 0 < k <= right - left + 1:
        index = parition(lst, left, right)
        if index - left == k - 1:
            return lst[index]
        if index - left > k - 1:
            return quick_select(lst, left, index - 1, k)
        return quick_select(lst, index + 1, right, k - index + left - 1)
    print("Index out of bounds")

# Minimally verbose Quick Select
def quickselect(arr, k):
    if len(arr) == 1:
        return arr[0]
    pivot = arr[0]
    lows = [x for x in arr[1:] if x < pivot]
    highs = [x for x in arr[1:] if x > pivot]
    pivots = [x for x in arr if x == pivot]
    if k < len(lows):
        return quickselect(lows, k)
    elif k < len(lows) + len(pivots):
        return pivot
    else:
        return quickselect(highs, k - len(lows) - len(pivots))

# Sorting with 2 stacks
def sort_stack(stack):
    sorted_stack = []
    while stack:
        temp = stack.pop()
        while sorted_stack and sorted_stack[-1] > temp:
            stack.append(sorted_stack.pop())
        sorted_stack.append(temp)
    while sorted_stack:
        stack.append(sorted_stack.pop())
    return stack

# Multi-conditional Sorting
def sort_grade_then_name(x):
    length = len(x)
    i = 0
    for i in range(length):
        smallest = i
        for j in range(i+1, length):
            if x[smallest][1] > x[j][1]:
                smallest = j
            elif x[smallest][1] == x[j][1]:
                if x[smallest][0] > x[j][0]:
                    smallest = j
        x[i], x[smallest] = x[smallest], x[i]
    return x

# Cut Rod Recitation 10 DP
def cut_rod_dp(n, prices):
    max_price = [0] * (n + 1)
    for length in range(1, n + 1):
        for p in prices:
            if p <= length:
                max_price[length] = max(max_price[length], prices[p] + max_price[length - p])
    return max_price[n]

# Cut Rod Recursive
def cut_rod(n, prices):
    if n == 0:
        return 0
    max_val = float('-inf')
    for i in range(1, n + 1):
        if i < len(prices):
            max_val = max(max_val, prices[i] + cut_rod_simple(n - i, prices))
    return max_val

# Cut Rod Memoized
def cut_rod_mm(n, prices):
    seen = {}
    def cut_rod_xx(n, prices, baggage):
        if n in seen:
            return seen[n]
    
        if n <=0:
            return 0,()
        else:
            max_price = 0
            for p in prices: # this is to go through all the options of cutting (key, which is the length)
                if p <= n:
                    temp_price, temp_bag = cut_rod_xx(n-p, prices, ())
                    if max_price < (prices[p]+temp_price):
                        max_price = prices[p]+temp_price
                        baggage = (p,) + temp_bag
                    
            seen[n] = (max_price, baggage)
            return max_price, baggage
    return cut_rod_xx(n, prices, ())

# ET Numbers
def ET_number(num, mapping):
    if num == 0:
        return mapping[0]
    base = len(mapping)
    alien_num = ''
    count = 0
    while num > 0:
        alien_num = mapping[num%base] + alien_num
        num = num//base
        count += 1
    return alien_num

# Combination between dp and subtree traversal
def subtree_distance(tree):
    n = len(tree)
    result = [0] * n  
    for i in range(n - 1, -1, -1): 
        parent = tree[i]
        if parent != -1:
            result[parent] = max(result[parent], result[i] + 1)
    return result

# Sum along an n-dimensional matrix
def sum_along(axis, arr):
    def helper(axis, arr, depth):
        if depth == axis:
            result = arr[0]
            for i in range(1, len(arr)):
                result = add(result, arr[i])
            return result
        return [helper(axis, sub, depth + 1) for sub in arr]
    def add(a, b):
        if isinstance(a, list) and isinstance(b, list):
            return [add(a[i], b[i]) for i in range(len(a))]
        else:
            return a + b
    return helper(axis, arr, 0)

# Transpose Matrix
def transpose(m):
    if not m: 
        return []
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]

# Matrix Multiplication
def multiply_matrices(matrix1, matrix2):
    rows1 = len(matrix1)
    cols1 = len(matrix1[0])
    rows2 = len(matrix2)
    cols2 = len(matrix2[0])
    if cols1 != rows2:
        raise ValueError("Matrix dimensions are incompatible for multiplication.")
    result_matrix = [[0 for _ in range(cols2)] for _ in range(rows1)]
    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result_matrix[i][j] += matrix1[i][k] * matrix2[k][j]
    return result_matrix
def multiply(mat1, mat2):
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            val = 0
            for k in range(len(mat2)):
                val += mat1[i][k] * mat2[k][j]
            row.append(val)
        result.append(row)
    return result

# Finding Max Recursively you can also use "or" for bool checks.
def max_collatz_distance(n):
    if n == 1:
        return 1
    else:
        return max(collatz_distance(n), max_collatz_distance(n-1))

# Memoization Wrapper @memoize
def memoize(f):
    memo = {}
    def wrapper(*args):
        if args not in memo:
            memo[args] = f(*args)
        return memo[args]
    return wrapper
memoize_table = {}
def memoize(f, name):
    if name not in memoize_table:
        memoize_table[name] = {}
    table = memoize_table[name]
    def helper(*args):
        if args in table:
            return table[args]
        else:
            result = f(*args)
            table[args] = result
            return result
    return helper

# Find max adjacent (if negative value just use 0)
# dp[i] = max(dp[i-1], vi + dp[i-2])
def mySum(lst):
    x, y = 0, lst[0]
    for i in range(1, len(lst)):
        x, y = y, max(y, x + lst[i])
    return y

# Knapsack
# function(Weights, Values, n) where n is the capacity of knapsack
# dp[W][i] = max(dp[W][i-1], vi + dp[W-Wi][i-1])
# W = remaining capacity of knapsack
# Edge Case: if the weight is too heavy then we dont include
# Base Case: if number of items = 0 value is 0
# if capacity = 0, no items can be included so the value is 0
def knapsack(n, W, weights, values):
    dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], values[i-1] + dp[i-1][w - weights[i-1]]) # Row wise
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][W]
def knapsack(vals, weights, w):
    dp = {}
    def helper(item_idx, current_weight_capacity):
        tpl = (item_idx, current_weight_capacity)
        if tpl in dp:
            return dp[tpl]
        if item_idx <= 0 or current_weight_capacity <= 0:
            return 0
        current_item_weight = weights[item_idx - 1]
        current_item_value = vals[item_idx - 1]
        value_if_not_taken = helper(item_idx - 1, current_weight_capacity)
        value_if_taken = 0
        if current_item_weight <= current_weight_capacity:
            value_if_taken = current_item_value + helper(item_idx - 1, current_weight_capacity - current_item_weight)
        dp[tpl] = max(value_if_not_taken, value_if_taken)
        return dp[tpl]
    return helper(len(vals), w)

weights_example = [10, 20, 30]
values_example = [60, 100, 120]
W_example = 50

print(knapsack(values_example, weights_example, W_example))

# Alternating Recursion
def stackn_alt_elegant(n, pic1, pic2):
    if n == 0:
        return pic1
    current_pic = pic1 if n % 2 != 0 else pic2
    return stack_frac(1/n, current_pic, stackn_alt(n - 1, pic2, pic1))

# BFS (tree tuple input) (head, left, right)
def bfs(tree):
    if tree is None or not tree:
        return []
    queue = []
    queue.append(tree) 
    result = []
    while queue:
        current_node_tuple = queue.pop(0)
        node_value = current_node_tuple[0]
        result.append(node_value)
        left_child_subtree = current_node_tuple[1]
        right_child_subtree = current_node_tuple[2]
        if left_child_subtree is not None:
            queue.append(left_child_subtree)
        if right_child_subtree is not None:
            queue.append(right_child_subtree)
    return result
from collections import deque
def level_order_traversal(tree_tuple):
    result = []
    queue = deque([tree_tuple])
    while queue:
        node = queue.popleft()
        if not isinstance(node, tuple):
            result.append(node)
        else:
            left, val, right = node
            result.append(val)
            queue.append(left)
            queue.append(right)
    return result

tree = (1,(2,(4, None, None),None),(3,(5,None,None),None))
# Sequence:
#[(2, (4, None, None), None), (3, (5, None, None), None)]
#[(3, (5, None, None), None), (4, None, None)]
#[(4, None, None), (5, None, None)]
#[(5, None, None)]
#[]
#[1, 2, 3, 4, 5]

# Deep Reverse
def deep_reverse(lst):
    if lst == []:
        return []
    elif type(a[0]) == list:
        return deep_reverse(lst[1:]) + [deep_reverse(lst[0])]
    else:
        return deep_reverse(lst[1:]) + [lst[0]]
def deep_reverse(lst):
    if not isinstance(lst, list):
        return lst
    return [deep_reverse(item) for item in reversed(lst)]
def deep_reverse(tree):
    if tree is None:
        return None
    value, left, right = tree
    return (value, deep_reverse(right), deep_reverse(left))


# Prefix Infix
def prefix_infix(expr):
    stack = []
    for i in range(len(expr) - 1, -1, -1):
        token = expr[i]
        if token in ['+', '-', '*', '/']:
            operand1 = stack.pop()
            operand2 = stack.pop()
            result = "(" + str(operand1) + str(token) + str(operand2) + ")"
            stack.append(result)
        else:
            stack.append(token)
    return stack.pop()
def prefix_infix(a):
    stack = make_stack()
    # consuming the input symbol one by one in a
    for op in a:
        if op in ["*", "/", "+", "-"]:
            stack("push", str(op))
        elif stack("peek") in ["*", "/", "+", "-"]:
            stack("push", str(op))
        else:
            temp = "(" + stack("pop") + stack("pop") + str(op)+ ")"
            while stack("size")>0 and stack("peek") not in ["*", "/", "+", "-"]:
                temp = "(" + stack("pop") + stack("pop") + temp + ")"
            stack("push", temp)
    stack("print")
    while stack("size") > 1:
        back = stack("pop")
        front = stack("pop")
        stack("push", "(" + front + stack("pop") + back+ ")")
    return stack("pop")

# Infix to Postfix
def infix_to_postfix(expr):
    precedence = ['+', '-', '*', '/']
    output = []
    stack = []
    for token in expr:
        if token not in ['+', '-', '*', '/', '(', ')']:
            output.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()  # Remove '('
        else:  # Operator
            while (stack and stack[-1] != '(' and
                   precedence.index(token) <= precedence.index(stack[-1])):
                output.append(stack.pop())
            stack.append(token)
    while stack:
        output.append(stack.pop())
    return output

# Infix to Prefix
def infix_to_prefix(expr):
    # Reverse and swap brackets
    expr = expr[::-1]
    for i in range(len(expr)):
        if expr[i] == '(':
            expr[i] = ')'
        elif expr[i] == ')':
            expr[i] = '('
    postfix = infix_to_postfix(expr)
    return postfix[::-1]


# Interleave
def interleave(seq1, seq2):
    if not seq1:
        return seq2
    elif not seq2:
        return seq1
    else:
        return [seq1[0], seq2[0]] + interleave(seq1[1:], seq2[1:])

# Power Set
def power_set(a):
    if a == []:
        return [[]]
    else:
        result1 = power_set(a[1:])
        result2 = list(map(lambda x: x+[a[0]], result1))
        return result1 + result2
def power_set(lst):
    if not lst:
        return [[]]
    head = lst[0]
    tail = lst[1:]
    sub_power_set = power_set(tail)
    subsets_with_head = [[head] + subset for subset in sub_power_set]
    return sub_power_set + subsets_with_head
def power_set(a):
    result = [[]]
    for item in a:
        new_subsets = []
        for subset in result:
            new_subsets.append(subset + [item])
        result += new_subsets
    return result
def power_set(a):
    if a == []:
        return [[]]
    rest = power_set(a[1:])
    with_first = []
    for subset in rest:
        with_first.append([a[0]] + subset)
    return rest + with_first

print(power_set([1,2,3]))

# Power of 2
def is_power_of_two(n):
    if n <= 0:
        return False
    while n % 2 == 0:
        n = n // 2
    return n == 1

print(is_power_of_two(9))

# ET Numbers
def dec_to_base(dec, base):
    if dec == 0:
        return "0"
    result = ""
    while int(dec) > 0:
        remainder = dec % base
        result = str(remainder) + result 
        dec //= base 
    return int(result)
def base_to_dec(num_in_base, base):
    if num_in_base == 0:
        return 0
    decimal_value = 0
    power = 0
    while num_in_base > 0:
        digit = num_in_base % 10 
        decimal_value += digit * (base ** power)
        num_in_base //= 10 
        power += 1
    return decimal_value

# Multi-conditional Sorting
def should_come_before(student1, student2):
    if student1[1] < student2[1]: # Compare age
        return True
    if student1[1] > student2[1]:
        return False
    return student1[2] > student2[2] # Compare score if age is equivalent

def insertion_sort_students(students):
    for i in range(1, len(students)):
        current_student = students[i]
        j = i - 1
        while j >= 0 and not should_come_before(students[j], current_student):
            students[j + 1] = students[j]
            j -= 1
        students[j + 1] = current_student

# Count Islands
def count_islands(grid):
    if not grid:
        return 0
    rows = len(grid)
    cols = len(grid[0])
    seen = [[False for _ in range(cols)] for _ in range(rows)]
    def bfs(i, j):
        q = [(i, j)]
        seen[i][j] = True
        while q:
            x, y = q.pop(0)
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols: # Within boundaries of grid
                    if grid[nx][ny] == 1 and not seen[nx][ny]: # Is island
                        seen[nx][ny] = True
                        q.append((nx, ny))
    count = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1 and not seen[i][j]:
                bfs(i, j)
                count += 1
    return count

# In order traversal
def in_order_traversal(tree_tuple):
    if not isinstance(tree_tuple, tuple):
        return [tree_tuple]
    left_subtree = tree_tuple[0]
    node_value = tree_tuple[1]
    right_subtree = tree_tuple[2]
    return in_order_traversal(left_subtree) + [node_value] + in_order_traversal(right_subtree)

# Post order traversal
def post_order_traversal(tree_tuple):
    if not isinstance(tree_tuple, tuple):
        return [tree_tuple]
    left_subtree = tree_tuple[0]
    node_value = tree_tuple[1]
    right_subtree = tree_tuple[2]
    left_result = post_order_traversal(left_subtree)
    right_result = post_order_traversal(right_subtree)
    return left_result + right_result + [node_value]

# Pre order traversal
def pre_order_traversal(tree_tuple):
    if not isinstance(tree_tuple, tuple):
        return [tree_tuple]
    left_subtree = tree_tuple[0]
    node_value = tree_tuple[1]
    right_subtree = tree_tuple[2]
    return [node_value] + pre_order_traversal(left_subtree) + pre_order_traversal(right_subtree)

# Binary Numbers
def next_idx(idx, shape):
    idx[-1] += 1
    for i in reversed(range(len(idx))):
        if idx[i] >= shape[i]:
            idx[i] = 0
            if i - 1 >= 0:
                idx[i - 1] += 1
            else:
                return None
    return idx

# Path
def path_sum(tree, targetSum):
    if tree is None:
        return False
    val, left, right = tree
    if left is None and right is None:
        return val == targetSum
    return path_sum(left, targetSum - val) or path_sum(right, targetSum - val)

# Recursive Indexing with Baggage
def path_sum(tree_list, targetSum):
    def dfs(index, current_sum):
        if index >= len(tree_list) or tree_list[index] is None:
            return False
        current_sum += tree_list[index]
        left = 2 * index + 1
        right = 2 * index + 2
        if (left >= len(tree_list) or tree_list[left] is None) and \
           (right >= len(tree_list) or tree_list[right] is None):
            return current_sum == targetSum
        return dfs(left, current_sum) or dfs(right, current_sum)
    return dfs(0, 0)

# Sum root to leaf and return list
def root_sum(tree):
    def helper(subtree, summation):
        if subtree is None:
            return []
        left, node, right = subtree
        if node is None: # Edge case if node is None which shouldnt be.
            return []
        summation += node
        if left is None and right is None:
            return [summation]
        return helper(left, summation) + helper(right, summation)
    return helper(tree, 0)

# DFS
def dfs_stack(tree):
    if tree is None:
        return []
    stack = [tree]
    result = []
    while stack:
        current = stack.pop()
        if current is None:
            continue
        value, left, right = current
        result.append(value)
        stack.append(right)
        stack.append(left)
    return result

class A:
    def method(self):
        print("A's method")
class B(A):
    def method(self):
        print("B's method")
        super().method()
class C(A):
    def method(self):
        print("C's method")
        super().method()
class D(B, C):
    def method(self):
        print("D's method")
        super().method()
# d = D()
# d.method() # Output: D>B>C>A

def rotate_matrix_90(mat):
    result = [[0] * len(mat) for _ in range(len(mat[0]))]
    for i in range(len(mat[0])):
        for j in range(len(mat)):
            result[i][j] = mat[len(mat) - 1 - j][i]
    return result

def create_symmetric(n):
    if n == 0:
        return None
    def build(val):
        if val == n:
            return [val, None, None]
        left = build(val + 1)
        right = build(val + 1)
        return [val, left, right]
    return build(1)

def construct(inorder, preorder):
    if not preorder or not inorder:
        return None
    root = preorder[0]
    root_index_in_inorder = inorder.index(root)
    left_inorder = inorder[:root_index_in_inorder]
    right_inorder = inorder[root_index_in_inorder + 1:]
    left_preorder = preorder[1:1 + len(left_inorder)]
    right_preorder = preorder[1 + len(left_inorder):]
    left_subtree = construct(left_inorder, left_preorder)
    right_subtree = construct(right_inorder, right_preorder)
    return [root, left_subtree, right_subtree]

# Recursion + DP + Memoization
dic = {}
def C_memo ( n ):
    if n<=1:
        return 0
    if dic.get(n, -1) != -1:
        return dic.get(n, 0)
    else:
        if dic.get(n//2, -1) == -1:
            dic.update( {n//2: C_memo(n//2)})
        if dic.get(n-1, -1) == -1:
            dic.update( {n-1: C_memo(n-1)})
    dic.update({n: n + dic.get(n//2,0) + dic.get(n-1,0)})
    return dic.get(n, 0)

# Deep Copy
def deep_copy(lst):
    if lst == []:
        return []
    elif type(lst[0]) == int:
        return [lst[0]] + deep_copy(lst[1:])
    elif isinstance(lst[0], list):
        return [deep_copy(lst[0])] + deep_copy(lst[1:])

# Dynamic Programming Cheapest Path Without Table
def cheapest_path(m, n, blocked):
    if (m, n) in blocked:
        return 4 * (m + n + 2)
    elif m == 0 and n == 0:
        return 0
    elif n > m or m < 0 or n < 0:
        return 4 * (m + n + 2)
    else:
        return min(
            1 + cheapest_path(m - 1, n, blocked),
            3 + cheapest_path(m - 1, n - 1, blocked),
            1 + cheapest_path(m, n - 1, blocked)
        )

# Deep Sum
def deep_sum(lst):
    total = 0
    for item in lst:
        if isinstance(item, int):
            total += item
        elif isinstance(item, list):
            total += deep_sum(item)
    return total

# Invert A Binary Tree
def invert(tree):
    if tree is None:
        return None
    value, left, right = tree
    inverted_left = invert(right)
    inverted_right = invert(left)
    return [value, inverted_left, inverted_right]
def invert(tree):
    if tree is None:
        return None
    stack = [tree]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        value, left, right = node
        node[1], node[2] = right, left
        stack.append(node[1])
        stack.append(node[2])
    return tree

print(list(map(lambda x, y: x+y, (1,2,3), (4,5,6))))

# Unbalanced to Balanced
def balance_tree(tree):
    if not tree:
        return None
    def get_values(t):
        if not t:
            return []
        root, left, right = t
        return get_values(left) + [root] + get_values(right)
    def build_balanced(values):
        if not values:
            return None
        mid = len(values) // 2
        return (values[mid], 
                build_balanced(values[:mid]), 
                build_balanced(values[mid+1:]))
    values = get_values(tree)
    return build_balanced(values)
def balance_tree(tree):
    if not tree: return None
    nodes = []
    queue = [tree]
    while queue:
        node = queue.pop(0)
        if node:
            nodes.append(node[0])
            queue.extend([node[1], node[2]])
    def build(arr, i=0):
        if i >= len(arr):
            return None
        left_child = build(arr, 2*i + 1)
        right_child = build(arr, 2*i + 2)
        return (arr[i], left_child, right_child)
    return build(nodes)
def balance_bst(tree):
    def in_order_traversal(node):
        if not node:
            return []
        return in_order_traversal(node[1]) + [node[0]] + in_order_traversal(node[2])
    sorted_values = in_order_traversal(tree)
    def build_balanced_tree(values):
        if not values:
            return None
        mid_index = len(values) // 2
        root_value = values[mid_index]
        left_subtree = build_balanced_tree(values[:mid_index])
        right_subtree = build_balanced_tree(values[mid_index + 1:])
        return (root_value, left_subtree, right_subtree)
    return build_balanced_tree(sorted_values)

# Track Traversal Path
def search_tree(tree, target, path=[]):
    if not tree:
        return False, path
    root, left, right = tree
    current_path = path + [root]
    if root == target:
        return True, current_path
    found, left_path = search_tree(left, target, current_path)
    if found:
        return True, left_path
    found, right_path = search_tree(right, target, current_path)
    if found:
        return True, right_path
    return False, []

# Edit Distance Leetcode
def edit_distance(word1, word2):
    dp = [[0 for _ in range(len(word1) + 1)] for _ in range(len(word2) + 1)]
    for i in range(len(word1) + 1):
        dp[0][i] = i
    for j in range(len(word2) + 1):
        dp[j][0] = j
    for i in range(len(word2) + 1):
        for j in range(len(word1) + 1):
            if word2[i-1] == word1[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                insert = dp[i-1][j] + 1
                delete = dp[i][j-1] + 1
                replace = dp[i-1][j-1] + 1
                dp[i][j] = min(insert, delete, replace)
    return dp[len(word2)][len(word1)]






























