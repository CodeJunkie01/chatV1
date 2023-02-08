import { env } from "../../../env/server.mjs";
import { z } from "zod";
import openai from "../external/openAI";
import { createTRPCRouter, protectedProcedure } from "../trpc";
import { spawn } from "child_process";
import type { PrismaClient } from "@prisma/client";
import { writeFile } from "fs";
export const openAI = createTRPCRouter({
	getAiselData: protectedProcedure.query(async ({ctx})=> await ctx.prisma.classifiedEmbedding.findMany()),
  getAnswer: protectedProcedure
    .input(z.object({ inputString: z.string() }))
    .mutation(async ({ input }) => {
      let response;
      if (env.OPENAI_ENABLED) {
        response = await openai.createCompletion({
          model: "text-curie-001",
          prompt: input.inputString,
          max_tokens: 500,
          temperature: 0.95,
        });
        const text = response?.data?.choices[0]?.text || "No answer found";
        if (text[0] === "?" || text[0] === ".") {
          return { answer: text.slice(1) };
        }
        return {
          answer: text,
        };
      }
      return { answer: "OpenAI is disabled" };
    }),
  getEmbedding: protectedProcedure
    .input(z.object({ inputString: z.string() }))
    .mutation(async ({ input }) => {
      let response;
      if (env.OPENAI_ENABLED) {
        response = await openai.createEmbedding({
          model: "text-embedding-ada-002",
          input: input.inputString,
        });
        console.log("data", response.data.data);
        return {
          embedding:
            response?.data.data[0]?.embedding.toString() ||
            "No embedding found",
        };
      }
      return { embedding: "OpenAI is disabled" };
    }),
  executePyScript: protectedProcedure
    .input(z.object({ queryX: z.string(), queryY: z.string() }))
    .mutation(async ({ ctx, input }) => {
      console.log("here");
      const { embedding: vectorX } = await getVectorForQuery({
        queryDefault: input.queryX,
        prisma: ctx.prisma,
      });
      const { embedding: vectorY } = await getVectorForQuery({
        queryDefault: input.queryY,
        prisma: ctx.prisma,
      });
      if (!vectorX || !vectorY) {
        throw new Error("No vector found");
      }
      const process = spawn("python3", [
        "./src/server/api/external/pyScript.py",
        vectorX.toString(),
        vectorY.toString(),
      ]);
      const pyPromise = new Promise((resolve) => {
        console.log("here4");
        //process.stderr.pipe(process.stdout);
        process.stdout.on("data", function (data: { cosine: string }) {
          console.log("data", data.toString());
          const x = data.toString().replaceAll("'", '"');
          // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
          resolve(JSON.parse(x).cosine || "No answer found");
        });
      });
      return { answer: await pyPromise };
    }),
  getSecretMessage: protectedProcedure.query(() => {
    return "you can now see this secret message!";
  }),
  getAllClasses: protectedProcedure.query(async ({ ctx }) => {
    const allClasses = await ctx.prisma.classifiedEmbedding.findMany({
      select: {
        class: true,
      },
    });
    const classes = <string[]>[];
    allClasses.forEach((x) => {
      if (!classes.includes(x.class)) {
        classes.push(x.class);
      }
    });
    return classes;
  }),
  getDataFromAisel: protectedProcedure
    .input(z.object({ baseUrl: z.string(),startIndex: z.string() }))
    .mutation(async ({ ctx, input }) => {
		const index = Number(input.startIndex);
      await getDataFromAisel({ urlBase: input.baseUrl, prisma: ctx.prisma, startIndex: index });
      return { message: "Data fetched" };
    }),
  classifyText: protectedProcedure
    .input(z.object({ text: z.string(), selectedClasses: z.array(z.string()) }))
    .mutation(async ({ input, ctx }) => {
      const { embedding } = await getVectorForQuery({
        queryDefault: input.text,
        prisma: ctx.prisma,
      });
      if (!embedding) {
        throw new Error("No vector found");
      }
      const allClasses = await ctx.prisma.classifiedEmbedding.findMany({
        where: {
          class: {
            in: input.selectedClasses,
          },
        },
        include: {
          Embedding: true,
        },
      });
      const data = allClasses.map((x) => {
        return `${x.class}; [${x.Embedding.embedding.toString()}]`;
      });
      const dataWithColumnNames = ["classes;embedding", ...data];
      writeFile(
        "src/server/csv-data/data.csv",
        dataWithColumnNames.join("\n"),
        (err) => {
          if (err) console.log(err);
        }
      );

      const process = spawn("python3", [
        "./src/server/api/external/randomForest.py",
        embedding.toString(),
      ]);
      const pyPromise = new Promise((resolve) => {
        console.log("here4");
        process.stderr.on("data", function (data: string) {
          console.log("data", data.toString());
        });
        process.stdout.on("data", function (data: { result: string }) {
          console.log("data", data.toString());
          const x = data.toString().replaceAll("'", '"');
          // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
          resolve(x || "No answer found");
        });
      });
      return { answer: await pyPromise };
    }),
});

const getVectorForQuery = async ({
  queryDefault,
  prisma,
}: {
  queryDefault: string;
  prisma: PrismaClient;
}) => {
  const query = queryDefault.replaceAll("\n", " ");
  let embedding = await prisma.embedding.findFirst({
    where: {
      query,
    },
  });
  if (!embedding) {
    const response = await openai.createEmbedding({
      model: "text-embedding-ada-002",
      input: query,
    });
    embedding = await prisma.embedding.create({
      data: {
        query,
        embedding: response?.data.data[0]?.embedding,
      },
    });
  }
  return embedding;
};

const getDataFromAisel = async ({
  urlBase,
  classString,
  prisma,
  startIndex,
}: {
  urlBase: string;
  classString?: string;
  prisma: PrismaClient;
  startIndex: number;
}) => {
  let moreData = true;
  let index = startIndex;
  while (moreData) {
    let localClassString = classString;
	let keywords = "";
    console.log("Fetching data from " + urlBase + "/" + index.toString());
    const response = await fetch(urlBase + "/" + index.toString());
    if (response.ok) {
      const data = await response.text();
      if (!localClassString) {
        const keywordMeta = data.match(/<meta name="keywords" content="[^>]*/);
        if (!keywordMeta) {
          continue;
        }
        keywords = keywordMeta[0]
          ?.replace('<meta name="keywords" content="', "")
          .replace(/"/g, "")
          .toLowerCase();
        console.log("keywords", keywords);
        const keywordsArray = keywords.split(", ");
        console.log("keywordsArray", keywordsArray);
        const keywordList = [
          "artificial intelligence",
          "machine learning",
          "deep learning",
          "neural network",
          "deep neural network",
          "deep neural networks",
          "advanced analytics",
          "data mining",
          "natural language processing",
          "nlp",
          "large language model",
          "llm",
          "large language models",
          "text mining",
        ];
        if (
          keywordList.some((keyword) =>
            keywords.includes(keyword.toLowerCase())
          )
        ) {
          localClassString = "artificial intelligence";
        } else {
          localClassString = "no artificial intelligence";
        }
      }
      const descriptionMeta = data.match(
        /<meta name="description" content="[^>]*/
      );
      if (!descriptionMeta) {
        continue;
      }
      const description = descriptionMeta[0]
        ?.replace('<meta name="description" content="', "")
        .replace(/"/g, "");
      const embedding = await getVectorForQuery({
        prisma,
        queryDefault: description,
      });
      await prisma.classifiedEmbedding.upsert({
        where: {
          embeddingId_class_unique: {
            embeddingId: embedding.id,
            class: localClassString,
          },
        },
        update: {},
        create: {
			keywords: keywords,
			url: urlBase + index,
          class: localClassString,
          Embedding: { connect: { id: embedding.id } },
        },
      });
      console.log(index.toString() + ": ", description);
      console.log("classified as: ", localClassString);
      console.log("\n");
    } else {
      moreData = false;
      console.log("No more data");
    }
    index++;
  }
};
