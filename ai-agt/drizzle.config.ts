import { env } from "@/data/env/server" 
import { defineConfig } from "drizzle-kit" // need drizzle-kit version v0.20 and up

export default defineConfig({
    out: "./src/drizzle/migrations",
    schema: "./src/drizzle/schema.ts",
    dialect: "postgresql",
    // strict and verbose are true to make drizzel to confirm with user all migrations, helps with security
    strict:true,
    verbose:true,

    // docker information
    dbCredentials: {
        password: env.DB_PASSWORD,
        user: env.DB_USER,
        database: env.DB_NAME,
        host: env.DB_HOST,
        ssl: false // ssl is false because we are doing this locally
    }
})