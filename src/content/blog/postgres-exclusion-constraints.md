---
author: Dan Clayton
pubDatetime: 2024-10-20T17:30:00
title: Preventing Overlapping Data in PostgreSQL - What Goes Into an Exclusion Constraint
postSlug: overlapping-data-postgres-exclusion-constraints
featured: false
draft: false
description: How can you prevent overlapping date ranges? Exclusion constraints do it at the database layer
---

Yesterday I knew nothing about PostgreSQL exclusion constraints. It turns out they are a really good way to prevent overlapping data in your database.

It's really common to have data, especially dates, that shouldn't overlap: assigning hotel rooms to guests; issuing equipment to employees or even defining regions on a map. Lots of people write application logic and validation code to prevent inconsistencies in their database, but if we can push this down onto the database layer, we can write less code. Less code is fewer tests to write, and probably fewer bugs. If you understand the benefits of unique or foreign key constraints, you'll like this too.

There are quite a few different Postgres features that go into this, so I've tried to bring them together for if, like me, most of this is new to you.

## A short working example

If you're running a hotel, you don't want to assign a room to multiple guests for the same night. There must be thousands of unit/feature tests for this problem out there in hotel codebases. We can be more confident in this by enforcing the constraint at the database level and trusting that Postgres is good software (it really is). This example is adapted from [PostgreSQL's documentation](https://www.postgresql.org/docs/current/rangetypes.html#RANGETYPES-CONSTRAINT);

```sql
CREATE TABLE room_reservations (
    room_name text,
    check_in timestamp,
    check_out timestamp
);

CREATE EXTENSION btree_gist;

ALTER TABLE room_reservations ADD CONSTRAINT no_overlapping_reservations
    EXCLUDE USING gist (
        room_name WITH =,
        tsrange(check_in, check_out) WITH &&
    );

INSERT INTO room_reservations VALUES
    ('123', '2024-01-01 14:00', '2024-01-03 11:00)');
OK INSERT 0 1;

INSERT INTO room_reservations VALUES
    ('123', '2024-01-02 14:00', '2024-01-04 11:00)');
ERROR:  conflicting key value violates exclusion constraint "no_overlapping_reservations"
DETAIL:  Key (room_name, tsrange(check_in, check_out))=(123, ["2024-01-02 14:00:00","2024-01-04 11:00:00")) conflicts with existing key (room_name, tsrange(check_in, check_out))=(123, ["2024-01-01 14:00:00","2024-01-03 11:00:00")).
```

So as long as we have decided on a data structure and added a constraint like this, we can be sure that we have integrity in the database. No room can be booked more than once at the same time. You might still have some application code to check this, for example to provide better validation feedback to users, but you will be less reliant on good tests (or crossed fingers).

> You might get this error: `ERROR:  data type text has no default operator class for access method "gist"
HINT:  You must specify an operator class for the index or define a default operator class for the data type.`
> You probably didn't enable the _btree_gist_ extension (`CREATE EXTENSION btree_gist;`). I explain what this error actually means below.

The use cases for this go beyond room reservations:

- assigning physical objects for date ranges is a perfect use case
- for price lists, making sure you only have one price for each point in time
- Postgres can work with geometry, so you can force that regions don't overlap

## What goes into the constraint (how can you modify it?)

### Exclusion constraint basics

With an exclusion constraint you specify a set of comparisons, and the database will guarantee that they can't all be true when compared across any two rows. That sounds quite abstract. Another way to put it is that particular column values (or expressions based on those values) can't coexist in a table.

Like any constraint, you can create them at the same time as the table, or add them later. The most simple example might look at just one column:

```sql
CREATE TABLE projects (
  name text,
  EXCLUDE USING btree (name WITH =)
);

INSERT INTO projects VALUES ('Gemini');
OK INSERT 0 1;
INSERT INTO projects VALUES ('Apollo');
OK INSERT 0 1;

INSERT INTO projects VALUES ('Apollo');
ERROR:  conflicting key value violates exclusion constraint "projects_name_excl"
DETAIL:  Key (name)=(Apollo) conflicts with existing key (name)=(Apollo).
```

In this example, we have _excluded_ that comparing the `name` column `WITH` `=` returns true. In normal English, two rows can't have the same value for `name` when compared with the `=` operator.

In this instance, we've recreated the same functionality as a UNIQUE constraint, just with a bit more flexibility. Exclusion constraints are basically supercharged unique constraints. The power comes when we combine multiple columns and use more complicated data types.

The main difference with other indexes in Postgres is that you need to define _how_ to compare the rows. The `WITH` keyword tells Postgres how we want the rows to be compared.

### Postgres ranges

In our example at the top, we had `check_in` and `check_out` columns. This is such a common pattern than Postgres has a built in data type called `tsrange` which represents a timestamp range. It is part of the wider [Range Types](https://www.postgresql.org/docs/current/rangetypes.html) family. There are also numeric ranges, date ranges and timezone aware timestamp ranges.

The main benefit of using a range type is you can use the range operators to calculate containment, intersections, overlaps etc which is much easier and less error prone than messing with inequalities.

_Is 15 in the range defined by 10 and 20?_

```sql
SELECT int4range(10, 20) @> 15;
TRUE
```

_yes_

These range types can be stored directly in tables as the data type of a column, or you can use them as functions, to help with queries or define indexes.

> Postgres uses conventional defaults to the upper and lower bounds of ranges, the lower bound is included and upper bound excluded [10, 20). You can [customise the bounds behaviour](https://www.postgresql.org/docs/current/rangetypes.html#RANGETYPES-INCLUSIVITY).

### The overlap && operator

One of the [range operators](https://www.postgresql.org/docs/current/functions-range.html#RANGE-OPERATORS-TABLE) is the `&&` (overlap) operator. This is especially useful for our date or timestamp ranges, but you can use it elsewhere.

For example, do two geometric shapes overlap?

```sql
select box('(2,2),(0,0)') && box('(1,1), (0,0)');
TRUE
```

This might be especially useful if you're working with geospatial data (latitude, longitude) in [PostGIS](https://postgis.net/docs/geometry_overlaps.html).

And for timestamp ranges:

```sql
select
  tsrange('2024-01-01 14:00', '2024-01-02 11:00)')
  &&
  tsrange('2024-01-02 14:00', '2024-01-03 11:00)');
FALSE
```

### GiST (Generalised Search Tree)

In our working example at the top, we used the `gist` index when specifying the exclude constraint. What that?

When you create a primary key, or a unique constraint, Postgres will automatically create a b-tree index on the column (or group of columns) that you reference in the constraint. An index is needed to _back_ the constraint.

I won't attempt to explain b-trees, Postgres [has documentation](https://www.postgresql.org/docs/17/btree.html) and PlanetScale have [written a nice post about them](https://planetscale.com/blog/btrees-and-database-indexes).

What you need to know is that b-trees are used for scalar data types, which is what you typically have in a primary key or unique constraint.

If you want to index more complex data types, such as our range types, you can use a GiST. GiST supports all our range operators like containment and overlap. So if you are building an exclude constraint you'll normally want to define it with a GiST. If it only contains scalar data types you can define it with a b-tree but in practice that would be better as a regular UNIQUE constraint.

GiST was designed for complex data types, so the default implementation doesn't support using scalar data types in a GiST index.

If you try using a scalar data type in a normal GiST index, you'll get this error:

```
ERROR:  data type text has no default operator class for access method "gist"
HINT:  You must specify an operator class for the index or define a default operator class for the data type.
```

If you only have scalar data, just use a b-tree. If you have an index that needs both scalar and more complex data types, we can use a built-in Postgres extension that defines GiST 'operator classes' that implement b-tree like behaviour. This will do what the hint is suggesting.

Run `CREATE EXTENSION btree_gist;` which enables the [built-in extension](https://www.postgresql.org/docs/17/btree-gist.html).

### Putting it together for a constraint

When you define an index in Postgres, such as for a constraint (but it doesn't have to be), you are not limited to using raw column values.

You could build an index of lowercase titles for films which might help searching:

```sql
CREATE INDEX lower_title_idx ON films ((lower(title)));

SELECT * from films where lower(title) = 'star wars: episode iv - a new hope';
SELECT * from films where lower(title) like 'star wars:%';
```

You can include multiple columns in one index too. Postgres will use the index when you query the data in the same way (this is something to watch out for, make sure Postgres is using your indexes with the [EXPLAIN](https://www.postgresql.org/docs/current/using-explain.html) functionality).

In the same way when we define our exclude constraint we can apply the tsrange() function to calculate a range type, which then lets us compare two rows with the `&&` operator.

```sql
...
  EXCLUDE USING gist (
          room_name WITH =,
          tsrange(check_in, check_out) WITH &&
    );
```

And that is how we can prevent overlapping data at the database layer.
