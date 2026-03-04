export { default } from "next-auth/middleware";

export const config = {
  matcher: ["/dashboard/:path*", "/alerts/:path*", "/species/:path*", "/animals/:path*"],
};
