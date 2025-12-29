import { UserTable } from "@/drizzle/schema";
import { db } from "@/drizzle/db";
import { User } from "@clerk/nextjs/server";   
import { eq } from "drizzle-orm"; 

// Handles creating, inserting, updating, and deleting users in the database

// create user data from clerk user object 
export async function insertUser(data: typeof UserTable.$inferInsert) {
    // could be undefined, need to modify typescript config file => tsconfig.json
    const [newUser] = await db
    .insert(UserTable)
    .values(data)
    .returning()
    .onConflictDoUpdate({
        target: [UserTable.clerkUserId],
        set: data,
    }) // anytime you insert deuplicate clerikUserId, update the user info
    
    if (newUser == null) throw new Error("Failed to insert user")
    
    return newUser;
}

// update user by clerkUserId
export async function updateUser(
    {clerkUserId} : {clerkUserId: string}, 
    data: Partial< typeof UserTable.$inferInsert>
) {
    // could be undefined, need to modify typescript config file => tsconfig.json
    const [updatedUser] = await db
    .update(UserTable)
    .set(data)
    .where(eq(UserTable.clerkUserId, clerkUserId))
    .returning()
    
    if (updatedUser == null) throw new Error("Failed to update user")
    
    return updatedUser;
}

export async function deleteUser(
    {clerkUserId} : {clerkUserId: string}, 
    data: Partial< typeof UserTable.$inferInsert>
) {
    // could be undefined, need to modify typescript config file => tsconfig.json
    const [deletedUser] = await db
    .update(UserTable)
    .set({
        deletedAt: new Date(),
        email: "redacted@deleted.com",
        name: "Deleted User",
        clerkUserId: "deleted",
        imageUrl: null,
    })
    .where(eq(UserTable.clerkUserId, clerkUserId))
    .returning()
    
    if (deletedUser == null) throw new Error("Failed to delete user")
    
    return deletedUser;
}