import { relations } from "drizzle-orm";
import { pgTable, text, integer, pgEnum } from "drizzle-orm/pg-core";
import { id, createdAt, updatedAt } from "../schemaHelpers";
import { CourseProductTable } from "./courseProduct";

export const productStatuses = ["private", "public"] as const;
export type ProductStatus = (typeof productStatuses)[number];
export const productStatusEnum = pgEnum("product_status", productStatuses);

// WILL DELETE THIS LATER, BECAUSE THIS IS JUST FOR SELLING COURSES ON A LEARNING MANAGEMENT SYSTEM
export const ProductTable = pgTable("courses", {
    id,
    name: text().notNull(),
    description: text().notNull(),
    imageUrl: text().notNull(),
    priceInDollars: integer().notNull(),
    status: productStatusEnum().notNull().default("private"),
    createdAt, // stores in the local timezone of the user
    updatedAt, // can run javascript code to run a new update
});

export const ProductRelationships = relations(
    ProductTable, 
    ({many}) => ({
    // define relationships here
        courseProducts: many(CourseProductTable),
}))