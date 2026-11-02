import asyncio

from app.db.database import create_database_pool


async def main():

    pool = await create_database_pool()

    async with pool.acquire() as connection:

        result = await connection.fetchval(
            "SELECT current_database()"
        )

        print("Connected successfully!")
        print("Database:", result)

    await pool.close()


asyncio.run(main())