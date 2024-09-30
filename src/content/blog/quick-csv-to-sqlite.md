---
author: Dan Clayton
pubDatetime: 2024-09-30T17:45:00
title: Quick CSV to SQLite with DuckDB
postSlug: csv-to-sqlite-with-duckdb
featured: false
draft: false
description: Quickly creating SQLite databases with DuckDB, without writing schemas
---

I recently started building a web application using SQLite, and needed to seed the database with some existing data. As is often the case, this was supplied in a CSV. I have recently been building DuckDB into my toolset a lot more and thought this task should be an easy win. A lot of existing docs/blogs start with creating a table in SQLite and specifying schemas... I wanted something more quick and dirty.

## CSV to SQLite table as fast as possible (the tl;dr)

We start with our CSV file, shown here as a table.

> sodor.csv

| Name   | Type                 | Color  | Number |
| ------ | -------------------- | ------ | ------ |
| Thomas | Tank Engine          | Blue   | 1      |
| Gordon | Tender Engine        | Blue   | 4      |
| Percy  | Tank Engine          | Green  | 6      |
| James  | Mixed-Traffic Engine | Red    | 5      |
| Cranky | Crane                | Yellow |        |

Let's build a SQLite database with this as our first table. We don't care too much about the schema; this is for development, and if you're using SQLite, schemas are less important.

```bash
$ duckdb -c "ATTACH 'app.db' as sqlite_db (TYPE SQLITE); \
  CREATE TABLE sqlite_db.personnel AS SELECT * FROM 'sodor.csv';"
```

That's it.

Did it work? Let's use the SQLite shell tool to check it.

```sql
$ sqlite3 app.db
sqlite> select * from personnel;
Thomas|Tank Engine|Blue|1
Gordon|Tender Engine|Blue|4
Percy|Tank Engine|Green|6
James|Mixed-Traffic Engine|Red|5
Cranky|Crane|Yellow|

sqlite> .schema personnel
CREATE TABLE personnel("Name" VARCHAR, "Type" VARCHAR, Color VARCHAR, Number BIGINT);
```

It did! And DuckDB inferred that our `Number` column would be better as a BIGINT (okay INT would be more efficient but better than a VARCHAR).

Finally, we can actually make the shell command even shorter (but maybe that's harder to understand on first reading it).

```bash
$ duckdb -c "ATTACH 'app.db' (TYPE SQLITE); \
  CREATE TABLE app.personnel AS FROM 'sodor.csv';"
```

> In this shorter version, DuckDB uses the `app` in `app.db` to name the SQLite database we can create tables in like `app.personnel`.

/tldr

## How did that work?

Let's break that down into the different features of DuckDB that were useful here.

### CSV Autodetection

For me, [CSV auto detection](https://duckdb.org/docs/data/csv/auto_detection.html) is one of the best features in DuckDB. In this example, we are starting with the string 'sodor.csv', from which DuckDB is using the filename to work out to use `read_csv_auto` function. This function works out the delimiter, data types for the columns and pulls out the header names.

So we already have a SQL-like table by doing `select * from 'sodor.csv'`.

### Create Table As Select (CTAS)

[CTAS](https://duckdb.org/docs/sql/statements/create_table.html#create-table--as-select-ctas) is a feature, not exclusive to DuckDB, that allows us to create a table with the schema from a select statement. It's especially useful in this situation, as we already have a schema in a table-like object from the CSV auto detection (you can run `describe from 'sodor.csv'). So we don't need to respecify a table schema if we already have one suitable; obviously if you want more control over the database schema you can create a table 'like normal' and insert into that with any transformations you want to do.

### DuckDB SQLite Extension

DuckDB has extensions for [SQLite](https://duckdb.org/docs/extensions/sqlite.html), PostgreSQL, MySQL etc which allows us to work directly with external databases.

> DuckDB does this in an intelligent way, data isn't copied into the DuckDB process, but we still get the benefits of DuckDB's query engine. With Postgres it [uses the binary transfer mode](https://duckdb.org/2022/09/30/postgres-scanner.html#implementation).

You can attach multiple databases to a DuckDB session. In our case we want to attach a new SQLite database, and create a table with data in it.

```sql
ATTACH 'app.db' as sqlite_db (TYPE SQLITE);
```

By default a database is created in the native DuckDB format, but we can qualify what storage type we want.

We can change the current 'main' database with a `use` statement, like you can in MySQL. But we don't then we access tables with the namespace like `sqlite_db.t1`.

With SQLite, once attached, DuckDB will create the database file on disk, and as we perform operations they are written to disk.

## Taking more control

The main focus of this write up is in the one-liner shell CSV to SQLite table command. Using DuckDB CLI `-c` flag, commands are executed in non-interactive mode which is useful for simple tasks like this or inside pipelines.

In interactive mode, we can many any changes we need if the defaults are not what we want.

Starting an interactive session:

```bash
$ duckdb
D select * from 'sodor.csv';
┌─────────┬──────────────────────┬─────────┬────────┐
│  Name   │         Type         │  Color  │ Number │
│ varchar │       varchar        │ varchar │ int64  │
├─────────┼──────────────────────┼─────────┼────────┤
│ Thomas  │ Tank Engine          │ Blue    │      1 │
│ Gordon  │ Tender Engine        │ Blue    │      4 │
│ Percy   │ Tank Engine          │ Green   │      6 │
│ James   │ Mixed-Traffic Engine │ Red     │      5 │
│ Cranky  │ Crane                │ Yellow  │        │
└─────────┴──────────────────────┴─────────┴────────┘
```

Let's assume we are really worried about storage space and want `Number` to be in smaller integer storage.

```bash
D select * from read_csv('sodor.csv', types={'Number':'TINYINT'});
┌─────────┬──────────────────────┬─────────┬────────┐
│  Name   │         Type         │  Color  │ Number │
│ varchar │       varchar        │ varchar │  int8  │
├─────────┼──────────────────────┼─────────┼────────┤
│ Thomas  │ Tank Engine          │ Blue    │      1 │
│ Gordon  │ Tender Engine        │ Blue    │      4 │
│ Percy   │ Tank Engine          │ Green   │      6 │
│ James   │ Mixed-Traffic Engine │ Red     │      5 │
│ Cranky  │ Crane                │ Yellow  │        │
└─────────┴──────────────────────┴─────────┴────────┘
```

Alternatively, we might want to manually create a table, either in DuckDB or in another database, and then move data into it. We could also use the [COPY](https://duckdb.org/docs/sql/statements/copy.html) command.

```sql
D CREATE TABLE personnel (nom varchar, nombre decimal);
D INSERT INTO personnel SELECT Name as nom, Number as nombre from 'sodor.csv';
D SELECT * FROM personnel;
┌─────────┬───────────────┐
│   nom   │    nombre     │
│ varchar │ decimal(18,3) │
├─────────┼───────────────┤
│ Thomas  │         1.000 │
│ Gordon  │         4.000 │
│ Percy   │         6.000 │
│ James   │         5.000 │
│ Cranky  │               │
└─────────┴───────────────┘
```

There's no limit to what we can do, DuckDB is very flexible with these operations, but the nice bit is that normally we can get pretty far with the defaults and very short commands.

## Aside: converting to Parquet

Converting to Parquet is even easier, I'm just adding it as Parquet is such a useful format and you can get massive benefits if you're only using CSV at the minute.

```bash
$ duckdb -c "COPY (FROM 'sodor.csv') TO 'sodor.parquet' (FORMAT PARQUET)"
```

It's worth looking at the Parquet options. The defaults are fine, but changing the compression settings might help if you have huge datasets. Do you want to optimise for speed or storage?
