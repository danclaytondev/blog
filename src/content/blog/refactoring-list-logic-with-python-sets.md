---
author: Dan Clayton
pubDatetime: 2023-11-24T17:00:00Z
title: Refactoring list logic using Python sets
postSlug: refactoring-list-logic-with-python-sets
featured: false
draft: true
description: Replacing loops and ifs with Python sets for a speedup and better readability.
---

A colleague and I recently revisited some Python code we had written because it was too slow. Originally we were comparing the contents of two lists, each with somewhere between 10 and 100 elements. People were then trying to use this web endpoint to compare lists with hundreds of thousands of elements, and it took far too long...time for a refactor!

## Table of contents

The original use case for this code was to compare to csv files, and check whether columns are present in both, and provide user feedback or not. There are lots of applications of comparing lists, so rather than flogging an abstract data engineering example, I will demonstrate this refactor with some baking!

## The original logic

Let us take an application where we want to compare two lists:

- What our recipe tells us we need to make a dish
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

We will just assume that we are consistent with strings to represent ingredients and ignore quantities of flour etc. In a real app, the list elements might be IDs which are comparable, or column headers in our case.

Let's first answer whether we can make the recipe at all, by working out what we are missing:

```python
missing_ingredients = [
    ingredient for ingredient in recipe if ingredient not in cupboard
]

if not missing_ingredients:
    # Equivalent to if len(missing_ingredients) == 0:
    # But PEP 8 reccomends using a list's truthyness
    print("You can make the recipe!")
else:
    print("You need to buy:", missing_ingredients)
```

This works fairly well. If the list of missing ingredients is empty, you can make the recipe!

> As an aside, we could have used booleans to find out if we are missing anything, which would be a performance boost\*:
>
> ```python
> missing_any_ingredients: bool = any(
>    ingredient not in cupboard for ingredient in recipe
> )
> ```
>
> \*_this is a Python generator_.
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

- Sort the lists, then compare them with Python equality or loop through and compare every element however you like
- Use our check above for missing items, check that is zero, then check the lists are the same length (this is convoluted but is relevant to this example)
- Using `collections.Counter` which counts the occurences of every element, in a dictionary you can then compare these `Counters`

**But there is a better\* way.**

\*it depends, better is subjective to your use case.

## Introducing Python sets
