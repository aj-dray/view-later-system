import { redirect } from "next/navigation";

import SettingsClient from "@/app/settings/SettingsClient";
import { getSession } from "@/app/_lib/auth";
import { getCurrentUser } from "@/app/_lib/user";

type PageProps = {
  searchParams?: Record<string, string | string[] | undefined>;
};

export default async function SettingsPage({ searchParams }: PageProps) {
  const session = await getSession();
  if (!session) {
    redirect("/login");
  }

  const user = await getCurrentUser().catch(() => null);

  const username = user?.username ?? session.username ?? null;

  const gmailStatusParam = searchParams?.gmail;
  const gmailStatus =
    typeof gmailStatusParam === "string" ? gmailStatusParam : null;

  return (
    <div className="flex h-full w-full flex-col overflow-auto bg-[#F0F0F0] items-center">
      <SettingsClient
        username={username}
        gmailStatus={gmailStatus}
      />
    </div>
  );
}
