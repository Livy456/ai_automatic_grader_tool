import { relations } from "drizzle-orm";
import { pgTable, text } from "drizzle-orm/pg-core";
import { id, createdAt, updatedAt } from "../schemaHelpers";
import { CourseProductTable } from "./courseProduct.js";
import {UserCourseAccessTable} from "./userCourseAccess.js";
// import { uuid } from "zod";

export const CourseTable = pgTable("courses", {
    id,
    name: text().notNull(),
    description: text().notNull(),
    createdAt, // stores in the local timezone of the user
    updatedAt, // can run javascript code to run a new update
});

export const CourseRelationships = relations(
    CourseTable, 
    ({one, many}) => ({
    // define relationships here
        courseProducts: many(CourseProductTable),
        UserCourseAccesses: many(UserCourseAccessTable)
}))