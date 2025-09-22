import Head from 'next/head'
import QueryBox from '../components/QueryBox'

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Head>
        <title>AI Notebook Assistant</title>
        <meta name="description" content="AI-powered assistant for your machine learning notes" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="py-8">
        <QueryBox />
      </main>

      <footer className="text-center py-8 text-gray-600">
        <p>AI Notebook Assistant - Powered by Firebase, FastAPI & Next.js</p>
      </footer>
    </div>
  )
}
