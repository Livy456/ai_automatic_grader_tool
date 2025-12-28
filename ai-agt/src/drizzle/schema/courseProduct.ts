import { pgTable, primaryKey, uuid } from "drizzle-orm/pg-core"
import { CourseTable } from "./course"
import { createdAt, updatedAt } from "../schemaHelpers"
import { relations } from "drizzle-orm/relations";
import { ProductTable } from "./product";

export const CourseProductTable = pgTable("course_products", {
    courseId: uuid()
    .notNull()
    .references(() => CourseTable.id, {onDelete: "restrict"}), // do not let someone delete a course if selling product
    productId: uuid()
    .notNull()
    .references(() => CourseTable.id, {onDelete: "cascade"}), // deletes product id if course id is gone
    createdAt,
    updatedAt
}, 
t => [primaryKey({columns: [t.courseId, t.productId] })]// throws an error if there is a duplicate courseId and productId combination
)

export const CourseProductRelationships = relations(
    CourseProductTable, 
    ({ one }) => ({
        course: one(CourseTable, {
            fields: [CourseProductTable.courseId],
            references: [CourseTable.id],
        }),

        product: one(ProductTable, {
            fields: [CourseProductTable.productId],
            references: [ProductTable.id],
        }),
    })
)