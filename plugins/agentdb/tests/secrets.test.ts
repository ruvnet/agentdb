import { describe, expect, test, tier } from 'claude-code/testing'

import { readOptions } from '../hooks/options'
import { looksRandom, scanInput, scanText } from '../hooks/secrets'

tier('user')

describe('scanText', () => {
  test('token shapes are found by name', () => {
    expect(scanText('AKIA' + 'IOSFODNN7EXAMPLQ')).toContain('aws access key')
    expect(scanText('sk-ant-' + 'api03-abcdefghijklmnopqrstuvwxyz012345')).toContain('anthropic or openai key')
    expect(scanText('postgres://admin:hunter2pass@db.internal/app')).toContain('url with credentials')
    expect(scanText('Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123')).toContain('bearer token')
  })

  test('a credential assignment needs a credential name and a random value', () => {
    expect(scanText('DB_PASSWORD=Zq8vT3mK9wX2')).toContain('credential assignment')
    expect(scanText('password=${DB_PASSWORD}')).toEqual([])
    expect(scanText('password: changeme123')).toEqual([])
    expect(scanText('timeout=30000ms retries=3')).toEqual([])
  })

  test('ordinary engineering notes pass', () => {
    expect(scanText('JWT refresh token rotation: issue a new refresh token on every use and revoke the old one.')).toEqual([])
    expect(scanText('HNSW beats brute force above 5k vectors; M=16, efSearch=64.')).toEqual([])
  })
})

describe('looksRandom', () => {
  test('references and placeholders are not values', () => {
    expect(looksRandom('process.env.API_KEY')).toBe(false)
    expect(looksRandom('<your-api-key>')).toBe(false)
    expect(looksRandom('xxxxxxxxxx')).toBe(false)
    expect(looksRandom('Zq8vT3mK9wX2')).toBe(true)
  })
})

describe('scanInput', () => {
  test('reads nested values, credential-named keys and name/value pairs', () => {
    expect(scanInput({ a: { b: ['note', 'ghp_' + 'Zq8vT3mK9wX2LpR7sNd4Yb6Hc1Jf5Ge0Qa3W'] } }).secrets).toContain('github token')
    expect(scanInput({ clientSecret: 'Zq8vT3mK9wX2LpR7' }).secrets).toContain('credential field')
    expect(scanInput({ key: 'api_key', value: 'Zq8vT3mK9wX2LpR7' }).secrets).toContain('credential field')
    expect(scanInput([['token', 'Zq8vT3mK9wX2LpR7']]).secrets).toContain('credential field')
    expect(scanInput({ key: 'project', value: 'Zq8vT3mK9wX2LpR7' }).secrets).toEqual([])
  })

  test('an input past a budget is reported oversize, not half read', () => {
    expect(scanInput({ list: Array.from({ length: 25_000 }, () => 1) }).oversize).toBe(true)
    expect(scanInput({ a: 'p'.repeat(1_100_000), b: 'p'.repeat(1_100_000) }).oversize).toBe(true)
    expect(scanInput({ a: 'lorem '.repeat(50_000) }).oversize).toBe(false)
  })
})

describe('readOptions', () => {
  test('defaults are the safe ones; unknown values fall back to them', () => {
    expect(readOptions(undefined)).toEqual({ secretGuard: 'enforce', allowDestructive: false })
    expect(readOptions({ secret_guard: 'loud', allow_destructive: 'yes' })).toEqual({ secretGuard: 'enforce', allowDestructive: false })
    expect(readOptions({ secret_guard: 'notify', allow_destructive: true })).toEqual({ secretGuard: 'notify', allowDestructive: true })
    expect(readOptions({ secret_guard: false })).toEqual({ secretGuard: 'off', allowDestructive: false })
  })
})
