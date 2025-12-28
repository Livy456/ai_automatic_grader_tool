import { relations } from "drizzle-orm";
import { pgTable, text, pgEnum, timestamp } from "drizzle-orm/pg-core";
import { createdAt, id, updatedAt } from "../schemaHelpers";
import { CourseProductTable } from "./courseProduct";

// User roles => user(student), instructor, admin
export const userAccessRoles = ["user", "instructor", "admin"] as const;
export type UserAccessRole = (typeof userAccessRoles)[number];
export const userAccessRoleEnum = pgEnum(
    "user_access_role", 
    userAccessRoles);

// makes a table with the different type of users
export const UserCourseAccessTable = pgTable("users_course_access", {
    id,
    clerkUserId: text().notNull(), // from clerk -> used for user management
    email: text().notNull(),
    name: text().notNull(),
    role: userAccessRoleEnum().notNull().default("user"),
    imageUrl: text(), // profile image
    deletedAt: timestamp({withTimezone: true}),
    createdAt, // stores in the local timezone of the user
    updatedAt // can run javascript code to run a new update
})
