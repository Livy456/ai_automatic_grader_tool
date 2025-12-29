import { SignedIn, SignedOut, SignInButton, UserButton } from "@clerk/nextjs";
import Link from "next/link";
import { ReactNode, Suspense } from "react";

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

// function Navbar() {
//   return <div className="p-4 bg-red-200">NAVBAR TEST</div>;
// }

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
                        <Link className="hover:bg-accent/10 flex items-center px-2" href="/courses">
                            My Courses
                        </Link>
                        <Link className="hover:bg-accent/10 flex items-center px-2" href="/purchases">
                            Purchase History
                        </Link>
                    </SignedIn>
                </Suspense>
            </nav>
        </header>
    )
}
