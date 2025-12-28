import { relations } from "drizzle-orm";
import { pgTable, text, integer, pgEnum, uuid } from "drizzle-orm/pg-core";
import { id, createdAt, updatedAt } from "../schemaHelpers";
import { CourseTable } from "./course";
import { LessonTable } from "./lesson";

export const courseSectionStatuses = ["private", "public"] as const;
export type CourseSectionStatus = (typeof courseSectionStatuses)[number];
export const courseSectionStatusEnum = pgEnum(
    "course_section_status", 
    courseSectionStatuses);

export const CourseSectionTable = pgTable("course_sections", {
    id,
    name: text().notNull(),
    description: text().notNull(),
    status: courseSectionStatusEnum().notNull().default("private"),
    order: integer().notNull(), // order from zero to infinity
    courseId: uuid()
    .notNull()
    .references(() => CourseTable.id, {onDelete: "cascade"}),
    createdAt, // stores in the local timezone of the user
    updatedAt, // can run javascript code to run a new update
});

export const CourseSectionRelationships = relations(
    CourseSectionTable,
    ({ many, one }) => ({
        course: one(CourseTable, {
            fields: [CourseSectionTable.courseId],
            references: [CourseTable.id],
        }),
        lessons: many(LessonTable),
    }),
)