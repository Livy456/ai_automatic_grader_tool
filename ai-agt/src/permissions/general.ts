import { UserRole } from "@/drizzle/schema"

export function canAccessAdminPage({ role }: {
    role: UserRole | undefined
}): boolean {
    return role === "admin" || role === "instructor"
}