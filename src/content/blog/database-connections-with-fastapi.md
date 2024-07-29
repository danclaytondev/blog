---
author: Dan Clayton
pubDatetime: 2024-07-28T17:30:00Z
title: Database Connections with FastAPI (Postgres)
postSlug: database-connections-with-fastapi
featured: false
draft: false
description: Sharing database connections between requests with FastAPI
---

When building an API, especially with something like FastAPI, it's rare that you don't need to connect to a database. Postgres in an incredibly popular, well tested, and powerful database - and it seems it's popularity is still growing. How you connect to that database can be an important component of your API service; opening a new connection to a Postgres server is much more costly than sending a query through an already established one. In general, you should keep in mind that connections aren't free performance wise. They are slow (relatively) to open, and having too many open is _very_ expensive for your postgres server, so don't forget to close them. I would say that if you're building a simple FastAPI server with small load, you don't need to worry about this too much - premature optimisation is bad!

There are different choices you can make, with none being suitable for every use case. A fairly simple and reliable pattern is to open a new connection at the start of an API request, and reuse that connection for every query you need to make, before closing the connection when that request has been returned. On the other end of the spectrum, a tool such as [pg_bouncer](https://www.pgbouncer.org/) maintains a pool of connections itself, and shares them between requests sent to pg_bouncer as a proxy.

Let's keep things simple to start with and build our way up to using a connection pool inside our FastAPI app.

### Most basic and verbose usage of psycopg

[psycopg](https://www.psycopg.org/psycopg3/) is the most used PostgreSQL adapter for Python, and provides an asyncio interface which means it works well with FastAPI using async.

The principles here will be very similar to other databases, such as sqlite, MySQL, or NoSQL databases.

For the most simple database access, we can open a new connection inside our path operation.

```python
@app.get("/visit/")
async def add_visit():
    # Open a connection
    conn = await psycopg.AsyncConnection.connect(conn_string)
    try:
        # use the connection
        cursor = conn.cursor()
        await cursor.execute("insert into visits(timestamp) values (now())")
        await cursor.close()
        # commit the transaction
        await conn.commit()
    except BaseException:
        await conn.rollback()
    finally:
        # We must always close our connection
        await conn.close()

    return {"message": "Visit logged"}
```

When psycopg is used like this, we must remember to always release our connections whether the query was successful or an exception was thrown, so a `finally` block is useful.

### Context managers

However, psycopg has built in support for context managers, which can simplify our code greatly and remove the need to remember to close connections.

```python
@app.get("/visit/")
async def add_visit():
    # Open a connection
    async with await psycopg.AsyncConnection.connect(conn_string) as conn:
        async with conn.cursor() as cursor:
            # Run our queries
            await cursor.execute("insert into visits(timestamp) values (now())")

    # We have left the block, so the context manager
    # has already committed and closed the connection

    return {"message": "Visit logged"}
```

> The biggest change and gotcha, is that psycopg automatically commits **open** transactions at the end of a block if there were no exceptions, so you don't need to call `commit`. You will need to use `rollback` if you don't want it committed. [See psycopg docs](https://www.psycopg.org/psycopg3/docs/basic/transactions.html)

### FastAPI Dependencies

Although context managers tidy up the code, we still need to open a connection in each operation, and it is hard to use throughout other parts of the application - we can do better.

FastAPI has a powerful dependency injection system which is best to learn from its [docs](https://fastapi.tiangolo.com/tutorial/dependencies/).

We can write dependencies, which can be injected into any path operation in our application. Dependencies can rely on each other, so we don't need any complex nesting code, and we can share resources like database connections throughout our application.

Lets go back to using psycopg without a context manager so we can see what is going on easily. As a first pass, we could write a dependency as follows:

```python
from fastapi import Depends, FastAPI
import psycopg

conn_string = "postgres://postgres@localhost"

app = FastAPI()

async def get_conn():
    return await psycopg.AsyncConnection.connect(conn_string)

@app.get("/visit/")
async def add_visit(conn=Depends(get_conn)):
    try:
        # use the connection
        async with conn.cursor() as cursor:
            # Run our queries
            await cursor.execute("insert into visits(timestamp) values (now())")
        # commit the transaction
        await conn.commit()
    except BaseException:
        await conn.rollback()
    finally:
        # We must always close our connection
        await conn.close()

    return {"message": "Visit logged"}
```

The problem with our setup is that we need to close our connection still, so we are back to using the manual method of closing connections and committing transactions.

How can we automatically close our connection with our FastAPI dependency? Using [dependencies with yield](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/).

By using a yield statement, FastAPI will inject the yielded object into the path operations, and then continue from the yield after the HTTP request has been returned.

```python
from fastapi import Depends, FastAPI
import psycopg

conn_string = "postgres://postgres@localhost"

app = FastAPI()

async def get_conn():
    try:
        conn = await psycopg.AsyncConnection.connect(conn_string)
        yield conn
        # after response is returned, commit the transaction
        await conn.commit()
    except BaseException:
        await conn.rollback()
    finally:
        # We must always close our connection
        await conn.close()

@app.get("/visit/")
async def add_visit(conn = Depends(get_conn)):

    async with conn.cursor() as cursor:
        # Run our queries
        await cursor.execute("insert into visits(timestamp) values (now())")

    return {"message": "Visit logged"}

```

In this setup, we have a connection ready to use inside each path operation, and we automatically commit, rollback, and close our connection as needed, without any code in our path operation. Each of our path operations are nice and simple now.

### Bringing it together

To make this even better, we can rewrite our FastAPI dependency to use the psycopg context manager like we did earlier before we had dependencies.

```python
from fastapi import Depends, FastAPI
import psycopg

conn_string = "postgres://postgres@localhost"

app = FastAPI()

async def get_conn():
    async with await psycopg.AsyncConnection.connect(conn_string) as conn:
	    yield conn

@app.get("/visit/")
async def add_visit(conn = Depends(get_conn)):

    async with conn.cursor() as cursor:
        # Run our queries
        await cursor.execute("insert into visits(timestamp) values (now())")

    return {"message": "Visit logged"}

```

This time when a response is returned, our dependency continues from the yield, which drops out of the context manager block (there is no more code to run after the yield), so psycopg automatically commits and closes the connection.

> A possible downside to this setup is that the postgres transaction is only committed after the response is returned and FastAPI is wrapping up the dependencies. This is likely to be fine for a lot of use cases and if you need to commit earlier then you can do so. Psycopg commits _open_ transactions for you at end of the context block.

### Add on: using a psycopg connection pool

With this dependency injection, it is now easy to swap to using a psycopg connection pool. We can open and close our pools with fine grained control using FastAPI events, and then take a connection from that pool in our dependency. At the end of each request, when the dependency tidies up, the connection will be released back to the pool.

```python
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
import psycopg_pool
import psycopg

conn_string = "postgres://postgres@localhost"

pool = psycopg_pool.AsyncConnectionPool(conn_string, open=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await pool.open()
    yield
    await pool.close()

app = FastAPI(lifespan=lifespan)

async def get_conn():
    async with await psycopg.AsyncConnection.connect(conn_string) as conn:
	    yield conn

@app.get("/visit/")
async def add_visit(conn = Depends(get_conn)):

    async with conn.cursor() as cursor:
        # Run our queries
        await cursor.execute("insert into visits(timestamp) values (now())")

    return {"message": "Visit logged"}

```

The code is available on [Github](https://github.com/danclaytondev/fastapi-database-connections).

### Beware how many connections you are opening

With Postgres, open idle connections are still consuming resources. This is fine if we open a fairly small pool; psycopg defaults to using a minimum size of 4 connections.

FastAPI uses uvicorn as its default server for development and basic deployment scenarios, however it can be setup to run with gunicorn rather than uvicorn and use multiple worker processes to optimise the performance on a single machine (making use of multiple cores).
In this scenario, because each worker process has independent memory, we will be setting up 4 independent connection pools, bringing us up to 16 connections at minimum. This will be the same if you use uvicorn in a container and horizontally scale your deployment.

It's hard to say how you should set this up, but you will want to be careful how many connections you have open.
