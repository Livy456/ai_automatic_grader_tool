import { Button } from "@/components/ui/button";
import { getCurrentUser } from "@/services/clerk";
import { SignedIn, SignedOut, SignInButton, UserButton } from "@clerk/nextjs";
import Link from "next/link";
import { ReactNode, Suspense } from "react";
import { canAccessAdminPage } from "@/permissions/general";

export default function StudentLayout({
    children
}: Readonly<{ children: ReactNode }>) {
    return (
        <>
            <Navbar />
            <main>{children}</main>
        </>
    )
}

function Navbar() {
    return (
        <header className="flex h-12 shadow 
        bg-background z-10">
            <nav className="flex gap-4 container"> 
                <Link className="mr-auto text-lg hover:underline 
                px-2 flex items-center" href="/">
                    AI AGT - Student
                </Link>
                <Suspense>
                    <SignedIn>
                        <AdminLink />
                        <Link className="hover:bg-accent/10 flex items-center px-2" href="/instructor">
                            Instructor
                        </Link>
                        <Link className="hover:bg-accent/10 flex items-center px-2" href="/courses">
                            My Courses
                        </Link>
                        <Link className="hover:bg-accent/10 flex items-center px-2" href="/purchases">
                            Purchase History
                        </Link>
                        <div>
                            <UserButton appearance={{
                                elements: {
                                    userButtonAvatarBox: { width: "100%", height: "100%" },
                                    },
                                }}
                            ></UserButton>

                        </div>
                    </SignedIn>
                </Suspense>
                <Suspense>
                    <SignedOut>
                        <Button className="self-center" asChild>
                            <Link href={"/sign-in"}> Sign In </Link>
                            <SignInButton> Sign In </SignInButton>
                        </Button>
                    </SignedOut>
                </Suspense>
            </nav>
        </header>
    )
}

async function AdminLink() {
    const user = await getCurrentUser();
    if(!canAccessAdminPage(user)) return null

    return (
        <Link className="hover:bg-accent/10 flex items-center px-2" href="/admin">
            Admin
        </Link>
    )
}
