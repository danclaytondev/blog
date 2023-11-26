---
author: Dan Clayton
pubDatetime: 2023-11-24T17:00:00Z
title: Refactoring list logic using Python sets
postSlug: refactoring-list-logic-with-python-sets
featured: false
draft: false
description: Replacing loops and ifs with Python sets for a speedup and better readability.
---

A colleague and I recently revisited some Python code we had written because it was too slow. Originally we were comparing the contents of two lists, each with somewhere between 10 and 100 elements. People were then trying to use this web endpoint to compare lists with hundreds of thousands of elements, and it was very slow...time for a refactor!

_Many people might know about sets already, and there are plenty of explanations of how to use them (and I definitely can't teach full-on set theory), but this post is more about the story and process of a refactor._

## Table of contents

## Context

The original use case for this code was to compare two csv files, and check whether columns are present in both, and provide user feedback if not. There are lots of applications of comparing lists, so rather than flogging an abstract data engineering example, I will demonstrate this refactor with some baking!

## The original logic

Let us take a dummy application where we want to compare two lists:

- What our recipe tells us we need to make a cake
- What we have in the cupboard

Most importantly, can we make the recipe with what we already have? But if not, we might want to find out what we need to buy from the corner shop, or, what we will have leftover.

Here is our recipe, and what we have in the cupboard:

```python
recipe = [
    "flour",
    "sugar",
    "eggs",
    "butter",
]

cupboard = [
    "flour",
    "sugar",
    "butter",
    "chocolate",
]
```

We will just assume that we are consistent with strings to represent ingredients and ignore quantities of flour etc. In a real app, the list elements might be IDs which are comparable, or column headers in our original case.

Let's first answer whether we can make the recipe at all, by working out what we are missing:

```python
missing_ingredients = [
    ingredient for ingredient in recipe if ingredient not in cupboard
]

if not missing_ingredients:
    # Equivalent to if len(missing_ingredients) == 0:
    # But PEP 8 recommends using a list's truthyness
    print("You can make the recipe!")
else:
    print("You need to buy:", missing_ingredients)
```

This works fairly well. If the list of missing ingredients is empty, you can make the recipe!

> As an aside, we could have used booleans\* to find out if we are missing anything, which would be more performant:
>
> ```python
> missing_any_ingredients: bool = any(
>    ingredient not in cupboard for ingredient in recipe
> )
> ```
>
> \*_this example is a Python generator_.
>
> However, this only tell us there is something missing, rather than being able to feedback what items are missing, which is often important.

What about any leftovers? We can write another list comprehension that is the inverse of the first:

```python
leftovers = [
    item for item in cupboard if item not in recipe
]

# leftovers -> ['chocolate']
```

Another common task is to compare the two lists to see if they are the same. This can be done in a number of ways:

- Sort the lists, then compare them with `==`, or loop through and compare every element however you like
- Use our check above for missing items, check that is zero length, then check the lists are the same length (this is convoluted but is relevant to this example)
- Using `collections.Counter` which counts the occurrences of every element, in a dictionary you can then compare these `Counters`

## What is wrong with this?

Not necessarily anything! Using a list comprehension like this is pretty Pythonic, and because of that, it's very readable.

`if ingredient not in cupboard` is very easy to understand on first read.

So if you are writing some logic like this, for recipes where we might have a dozen elements, this might be fine.

We hit some difficulty when both lists are long (which is often the case if you are comparing two similar lists).

Looking at the first list comprehension:

```python
missing_ingredients = [
    ingredient for ingredient in recipe if ingredient not in cupboard
]
```

we can workout what is going on. Let's assume `cupboard` and `recipe` are (roughly) the same length (n elements long), but contain different items.

In a list comprehension, we loop through each n elements in the list once and perform our logic, `recipe` is `n` elements long and the time complexity of the comprehension scales linearly with n. Inside the comprehension we run an if statement for every element in the outer comprehension. This if statement is `if ingredient (string) in cupboard (list)`. The Python `in` operator has to loop through a list to evaluate whether that list contains the string. This [is an O(n) operation](https://wiki.python.org/moin/TimeComplexity).

For the whole thing, we have a time [complexity](https://en.wikipedia.org/wiki/Big_O_notation) of O(n<sup>2</sup>). (As we have an O(n) inside an O(n)).

You won't notice that with small lists, but if they get big, it scales up very fast. We can do better!

## Using Python sets

I will leave a proper introduction of sets to [Python's documentation](https://docs.python.org/3/tutorial/datastructures.html#sets). They are an unordered collection with no duplicate items. We can take our lists, turn them into sets, and we unlock a wide range of benefits.

If you come from a mathematical background, you might reach to sets a bit sooner, as they use set theory.

The first of these is 'fast' membership testing.

```python
recipe_set = set(recipe)

'flour' in recipe_set # True
```

Sets, like dictionaries, use hashing to store their items. A hash of each item is computed, and recorded in a predictable location in memory (see [Hash table](https://en.wikipedia.org/wiki/Hash_table)). Working out if a table contains a hash is very fast, and doesn't increase with the number of items in the set, which means it is O(1) complexity.

Let's improve our code for faster testing:

```python
cupboard_set = set(cupboard)

missing_ingredients = [
    ingredient for ingredient in recipe if ingredient not in cupboard_set
]
```

With that simple change, the `missing_ingredients` has gone from O(n<sup>2</sup>) to O(n), which is a hefty speedup.

Whilst that is great, I actually think the better improvements come from the mathematical operations that sets offer. Unlike lists, we can use built in logic compute the mathematical relationship between two sets, which is not only much faster, but is arguably more readable (and less likely to have bugs).

The simplest of these is the difference between them using the minus operator. Our code becomes:

```python
# Items in recipe_set but not in cupboard_set
missing_ingredients = recipe_set - cupboard_set
# this is the same as
missing_ingredients = recipe_set.difference(cupboard_set)
```

Swapping the order around finds the leftovers:

```python
leftovers = cupboard_set - recipe_set
```

We can also compute the union, intersection, and symmetric difference of the sets. The symmetric difference is elements in either of the lists but not both (like XOR). There's a good chance you're familiar with these terms already (and you can pick it up if not), so using the built-in functions with these names will make your code very easy for others understand, and less error-prone.

Possibly one of the best is just comparing them for equality:

```python
a = set(['flour', 'butter'])
b = set(['butter', 'flour'])
a == b # True
```

will be true if each set's elements are in the other. If you keep in mind that turning a list into a set will remove duplicates, this is often the easiest way to compare lists.

All of your effort can be focused on working out which mathematical operation you need for your application, which is the best use of your brain as a software developer, rather than re-implementing logic that already exists in the language.

## Original code, refactored

Let's take a look at the original code, but using sets instead:

```python
recipe = set([
    "flour",
    "sugar",
    "eggs",
    "butter",
])

cupboard = set([
    "flour",
    "sugar",
    "butter",
    "chocolate",
])

missing_ingredients = recipe - cupboard # -> set(['eggs'])
leftovers = cupboard - recipe # -> set(['chocolate'])
```

The primary benefit for us was a _roughly_ 500x speedup (not properly benchmarked). We also got a big maintainability/readability improvement along the way.

At a most basic level, we could just validate for equality: `recipe == cupboard`, but with the difference operations, we can provide meaningful user feedback to the tune of: "You need to go to the shop to buy eggs".

I find it very rewarding to remove userland code and replace it with built-in functions.
