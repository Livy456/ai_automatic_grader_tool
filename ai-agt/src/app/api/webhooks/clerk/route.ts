import { WebhookEvent } from "@clerk/nextjs/server";
import { Webhook } from "svix";
import { headers } from "next/headers";
import { env } from "@/data/env/server";
import { insertUser, updateUser, deleteUser } from "@/features/users/db/users";
import { syncClerkUserMetadata } from "@/services/clerk";
// import { ClerkUserPublicMetadata } from "@/data/typeOverrides/clerk.d";

// based on clerk documentation for webhooks
export async function POST( req: Request) {
    const headerPayload = await headers()

    // gets information of who is sending the webhook from svix headers and when it was sent
    const svixId = headerPayload.get("svix-id")
    const svixTimestamp = headerPayload.get("svix-timestamp")
    const svixSignature = headerPayload.get("svix-signature")

    if (!svixId || !svixTimestamp || !svixSignature) {
        return new Response("Error occurred -- no svix headers", { status: 400 });
    }

    const payload = await req.json()
    const body = JSON.stringify(payload)
    //// 2) Get raw body (MUST be raw)
    // const body = await req.text(); // MIGHT NEED TO REPLACE THE ABOVE LINE WITH THIS LINE!!

    const wh = new Webhook(env.CLERK_WEBHOOK_SECRET)
    let event: WebhookEvent

    try{
        event = wh.verify(body, {
            "svix-id": svixId,
            "svix-timestamp": svixTimestamp,
            "svix-signature": svixSignature,
        }) as WebhookEvent
    } catch (error) {
        console.error("Error verifying webhook:", error)
        return new Response("Error occurred", { status: 400 })
    }

    // application specific logic to handle the event
    switch (event.type) {
        case "user.created":
        case "user.updated":
        {
            const email = event.data.email_addresses.find(
                email => email.id === event.data.primary_email_address_id
            )?.email_address // email is object so get the email address property

            const name = `${event.data.first_name} ${event.data.last_name}`.trim()

            if (email == null) return new Response("No email found", { status: 400 })
            if (name === "") return new Response("No name found", { status: 400 })
                
            if (event.type === "user.created") {
                const user = await insertUser({
                    clerkUserId: event.data.id,
                    email,
                    name,
                    imageUrl: event.data.image_url,
                    role: "user",

                })
                await syncClerkUserMetadata(user) // sync db id and role to clerk public metadata
            } else  {
                await updateUser({ clerkUserId: event.data.id }, {
                    email,
                    name,  
                    imageUrl: event.data.image_url,
                    role: event.data.public_metadata.role as "user" | "admin" | "instructor",
                })
            }
            break
        }  
        case "user.deleted":
        {
            if (event.data.id != null) {
                await deleteUser({ clerkUserId: event.data.id })
            }
            break   
        }
    }

    return new Response("", { status: 200 }) // lets clerk know we handled the webhook successfully
}

