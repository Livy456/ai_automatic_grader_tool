import { env } from "@/data/env/server"
import { drizzle } from "drizzle-orm/node-postgres"
import * as schema from "./schema"

// creates a database variable so you can access database from anywhere in the code
export const db = drizzle({
    schema,
    connection: {
        password: env.DB_PASSWORD,
        user: env.DB_USER,
        host: env.DB_HOST,
        database: env.DB_NAME,
    }
})
