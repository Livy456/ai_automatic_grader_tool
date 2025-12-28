import { relations } from "drizzle-orm";
import { pgTable, text, integer, pgEnum, uuid } from "drizzle-orm/pg-core";
import { id, createdAt, updatedAt } from "../schemaHelpers";
import { CourseTable } from "./course";
import { CourseSectionTable } from "./courseSection";
import { UserLessonCompleteTable } from "./userLessonComplete";

export const lessonStatuses = ["private", "public", "preview"] as const;
export type LessonStatus = (typeof lessonStatuses)[number];
export const lessonStatusEnum = pgEnum(
    "lesson_status", 
    lessonStatuses);

export const LessonTable = pgTable("lessons", {
    id,
    name: text().notNull(),
    description: text(),
    youtubeVideoId: text().notNull(), // Can replace this with the PATH video ID later or hosted platform
    status: lessonStatusEnum().notNull().default("private"),
    order: integer().notNull(), // order from zero to infinity
    sectionId: uuid()
    .notNull()
    .references(() => CourseSectionTable.id, {onDelete: "cascade"}),
    createdAt, // stores in the local timezone of the user
    updatedAt, // can run javascript code to run a new update
});

export const LessonRelationships = relations(
    LessonTable,
    ({ many, one }) => ({
        section: one(CourseSectionTable, {
            fields: [LessonTable.sectionId],
            references: [CourseSectionTable.id],
        }),
        UserLessonCompletes: many(UserLessonCompleteTable),
    })
)