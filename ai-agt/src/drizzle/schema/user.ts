import { relations } from "drizzle-orm";
import { pgTable, text, integer, pgEnum, uuid, timestamp } from "drizzle-orm/pg-core";
import { createdAt, id, updatedAt } from "../schemaHelpers";
import { UserCourseAccessTable } from "./userCourseAccess";

// User roles => user(student), instructor, admin
export const userRoles = ["user", "instructor", "admin"] as const;
export type UserRole = (typeof userRoles)[number];
export const userRoleEnum = pgEnum(
    "user_role", 
    userRoles);

// makes a table with the different type of users
export const UserTable = pgTable("users", {
    id,
    clerkUserId: text().notNull(), // from clerk -> used for user management
    email: text().notNull(),
    name: text().notNull(),
    role: userRoleEnum().notNull().default("user"),
    imageUrl: text(), // profile image
    deletedAt: timestamp({withTimezone: true}),
    createdAt, // stores in the local timezone of the user
    updatedAt // can run javascript code to run a new update
})

export const UserRelationships = relations(
    UserTable, 
    ({one, many}) => ({
    // define relationships here
        UserCourseAccesses: many(UserCourseAccessTable)
}))