import { UserRole } from "@clerk/types";

export {}

// make sure ClerkUserPublicMetadata includes our custom fields, making an interface
declare global {
    interface CustomJwtSessionClaims {
        dbId?: string;
        role?: UserRole;
    }

    interface ClerkUserPublicMetadata {
        dbId?: string;
        role?: UserRole;
    }
}