import { pgTable, primaryKey, uuid } from "drizzle-orm/pg-core";
import {createdAt, updatedAt} from "../schemaHelpers.js";
import { LessonTable } from "./lesson.js";
import { UserTable } from "./user.js";
import { relations } from "drizzle-orm";

export const UserLessonCompleteTable = pgTable(
    "user_lesson_complete", 
    {
        userId: uuid()
        .notNull()
        .references(() => UserTable.id, { onDelete: "cascade" }),
        lessonId: uuid()
        .notNull()
        .references(() => LessonTable.id, { onDelete: "cascade" }),
        createdAt, // stores in the local timezone of the user
        updatedAt // can run javascript code to run a new update
    },
    t => [primaryKey({columns: [t.userId, t.lessonId]})]
)

export const UserLessonCompleteRelationships = relations(
    UserLessonCompleteTable, 
    ({one}) => ({
        user: one(UserTable, {
            fields: [UserLessonCompleteTable.userId],
            references: [UserTable.id],
        }),
        lesson: one(LessonTable, {
            fields: [UserLessonCompleteTable.lessonId],
            references: [LessonTable.id],
        }),
    })
)