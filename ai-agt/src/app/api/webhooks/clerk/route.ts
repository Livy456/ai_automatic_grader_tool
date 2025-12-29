import { WebhookEvent } from "@clerk/nextjs/server";
import { Webhook } from "svix";
import { headers } from "next/headers";
import { env } from "@/data/env/server";

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

    const wh = new Webhook(env.CLERK_WEBHOOK_SECRET!)
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
}
