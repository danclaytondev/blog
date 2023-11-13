import type { Site, SocialObjects } from "./types";

export const SITE: Site = {
  website: import.meta.env.URL ?? "http://localhost:4321", // replace this with your deployed domain
  author: "Dan Clayton",
  desc: "A simple software blog.",
  title: "Dan Clayton's Blog",
  lightAndDarkMode: false,
  postPerPage: 5,
};

export const LOCALE = ["en-EN"]; // set to [] to use the environment default

export const LOGO_IMAGE = {
  enable: false,
  svg: true,
  width: 216,
  height: 46,
};

export const SOCIALS: SocialObjects = [
  {
    name: "Github",
    href: "https://github.com/danclaytondev/",
    linkTitle: ` ${SITE.author} on Github`,
    active: false,
  },
];
