#!/usr/bin/env python3
"""
RL Data Construction - Generate 2000 Prompts from Trending Frontend Demos

This script generates 2000 RL training prompts based on real task categories from
the existing RL dataset. Each prompt is expanded with different tech stacks.

Checklist Structure (20 items):
- ID 0-9: Category-specific items (10 items)
- ID 10-19: Universal items (10 items, shared across all categories)

Usage:
    python generate_from_trending_demos.py \
        --output rl_prompts_2000.jsonl \
        --target_count 2000
"""

import json
import os
import random
import argparse
from typing import List, Dict, Any


# =============================================================================
# Universal Checklist Items (ID 10-19) - Shared by ALL categories
# =============================================================================

UNIVERSAL_CHECKLIST = [
    {"id": 10, "title": "Can the code run directly?", "maxScore": 5,
     "description": "Verify code runs in browser without errors. Perfect: 5pts, minor issues: 3pts, won't run: 0pts."},
    {"id": 11, "title": "How robust is the code?", "maxScore": 5,
     "description": "Check handling of edge cases and errors. Robust: 5pts, basic: 3pts, fragile: 0pts."},
    {"id": 12, "title": "What is the code engineering quality?", "maxScore": 5,
     "description": "Check structure, naming, comments. High quality: 5pts, basic: 3pts, poor: 0pts."},
    {"id": 13, "title": "How aesthetically pleasing is the interface?", "maxScore": 5,
     "description": "Evaluate visual design. Beautiful: 5pts, acceptable: 3pts, ugly: 0pts."},
    {"id": 14, "title": "Is the interaction experience smooth?", "maxScore": 5,
     "description": "Check responsiveness and fluidity. Smooth: 5pts, acceptable: 3pts, laggy: 0pts."},
    {"id": 15, "title": "Are there innovative highlights?", "maxScore": 5,
     "description": "Check for creativity beyond requirements. Innovative: 5pts, standard: 3pts, none: 0pts."},
    {"id": 16, "title": "Is there functional redundancy?", "maxScore": 5,
     "description": "Check for unnecessary bloat. No redundancy: 5pts, minor: 3pts, severe: 0pts."},
    {"id": 17, "title": "Are code comments sufficient?", "maxScore": 5,
     "description": "Check documentation quality. Sufficient: 5pts, basic: 3pts, lacking: 0pts."},
    {"id": 18, "title": "How is performance?", "maxScore": 5,
     "description": "Evaluate speed and efficiency. Excellent: 5pts, acceptable: 3pts, poor: 0pts."},
    {"id": 19, "title": "Is compatibility good?", "maxScore": 5,
     "description": "Check cross-browser support. Good: 5pts, basic: 3pts, poor: 0pts."},
]


# =============================================================================
# Task Categories with Full Checklists (ID 0-9 are category-specific)
# =============================================================================

TASK_CATEGORIES = {
    "game_core": {
        "name": "Classic Games",
        "weight": 0.12,
        "tasks": [
            "Create a Snake game with neon visual style",
            "Build a Pac-Man clone with ghost AI",
            "Implement a Breakout/Arkanoid game with power-ups",
            "Create a Flappy Bird clone with score tracking",
            "Build a Pong game with two-player mode",
            "Develop a Space Invaders clone",
            "Create a Frogger-style road crossing game",
            "Build an Asteroids game with ship controls",
            "Implement a Candy Crush-style game",
            "Create a 2048 number puzzle game",
        ],
        "checklist": [
            {"id": 0, "title": "Is the core gameplay complete?", "maxScore": 5,
             "description": "Check main mechanics. Complete: 5pts, basic: 3pts, incomplete: 0pts."},
            {"id": 1, "title": "Is the game loop complete?", "maxScore": 5,
             "description": "Check start-play-end cycle. Complete: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 2, "title": "Is collision detection accurate?", "maxScore": 5,
             "description": "Check hit detection. Accurate: 5pts, basic: 3pts, broken: 0pts."},
            {"id": 3, "title": "Is game state management correct?", "maxScore": 5,
             "description": "Check score/lives tracking. Correct: 5pts, basic: 3pts, buggy: 0pts."},
            {"id": 4, "title": "Is visual feedback sufficient?", "maxScore": 5,
             "description": "Check feedback for actions. Sufficient: 5pts, basic: 3pts, lacking: 0pts."},
            {"id": 5, "title": "Is sound effect design appropriate?", "maxScore": 5,
             "description": "Check audio feedback. Appropriate: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 6, "title": "Is the frame rate stable?", "maxScore": 5,
             "description": "Check performance. Stable 60fps: 5pts, 30fps: 3pts, laggy: 0pts."},
            {"id": 7, "title": "Is the operation guidance clear?", "maxScore": 5,
             "description": "Check instructions. Clear: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 8, "title": "How is the replay value?", "maxScore": 5,
             "description": "Check replayability. High: 5pts, medium: 3pts, low: 0pts."},
            {"id": 9, "title": "Is the adaptability good?", "maxScore": 5,
             "description": "Check responsive design. Good: 5pts, basic: 3pts, broken: 0pts."},
        ]
    },
    "game_action": {
        "name": "Action Games",
        "weight": 0.06,
        "tasks": [
            "Implement a shooting game where players control a spaceship and shoot enemies",
            "Create a platformer game with a character that can jump and avoid obstacles",
            "Build a tower defense game with multiple tower types and enemy waves",
            "Implement a side-scrolling shooter with power-ups",
            "Create a zombie survival game with resource management",
            "Build a bullet hell game with complex enemy patterns",
            "Develop an endless runner with increasing difficulty",
            "Create a fighting game with combo attacks",
            "Build a racing game with obstacles",
            "Implement a rhythm action game",
        ],
        "checklist": [
            {"id": 0, "title": "Is action control smooth?", "maxScore": 5,
             "description": "Check control responsiveness. Smooth: 5pts, acceptable: 3pts, laggy: 0pts."},
            {"id": 1, "title": "Is the combat system complete?", "maxScore": 5,
             "description": "Check attack/defense mechanics. Complete: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 2, "title": "Is collision detection accurate?", "maxScore": 5,
             "description": "Check hit detection. Accurate: 5pts, basic: 3pts, broken: 0pts."},
            {"id": 3, "title": "Is game state management correct?", "maxScore": 5,
             "description": "Check score/lives/level tracking. Correct: 5pts, basic: 3pts, buggy: 0pts."},
            {"id": 4, "title": "Is visual feedback sufficient?", "maxScore": 5,
             "description": "Check feedback for actions. Sufficient: 5pts, basic: 3pts, lacking: 0pts."},
            {"id": 5, "title": "Is sound effect design appropriate?", "maxScore": 5,
             "description": "Check audio feedback. Appropriate: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 6, "title": "Is the frame rate stable?", "maxScore": 5,
             "description": "Check performance. Stable 60fps: 5pts, 30fps: 3pts, laggy: 0pts."},
            {"id": 7, "title": "Is the operation guidance clear?", "maxScore": 5,
             "description": "Check instructions. Clear: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 8, "title": "How is the replay value?", "maxScore": 5,
             "description": "Check replayability. High: 5pts, medium: 3pts, low: 0pts."},
            {"id": 9, "title": "Is the adaptability good?", "maxScore": 5,
             "description": "Check responsive design. Good: 5pts, basic: 3pts, broken: 0pts."},
        ]
    },
    "game_puzzle": {
        "name": "Puzzle Games",
        "weight": 0.04,
        "tasks": [
            "Create a match-3 puzzle game with cascading effects",
            "Build a Tetris clone with ghost piece preview",
            "Implement a Sudoku game with hint system",
            "Create a sliding puzzle with image support",
            "Build a word search game with timer",
            "Create a jigsaw puzzle with drag-and-drop",
            "Build a maze game with fog of war",
            "Implement a Tower of Hanoi with drag-and-drop",
            "Create a memory card matching game",
            "Build a nonogram/picross puzzle game",
        ],
        "checklist": [
            {"id": 0, "title": "Is the puzzle core mechanism complete?", "maxScore": 5,
             "description": "Check core puzzle logic. Complete: 5pts, basic: 3pts, broken: 0pts."},
            {"id": 1, "title": "Is the level design reasonable?", "maxScore": 5,
             "description": "Check difficulty progression. Reasonable: 5pts, basic: 3pts, poor: 0pts."},
            {"id": 2, "title": "Is collision detection accurate?", "maxScore": 5,
             "description": "Check piece placement. Accurate: 5pts, basic: 3pts, broken: 0pts."},
            {"id": 3, "title": "Is game state management correct?", "maxScore": 5,
             "description": "Check progress/score tracking. Correct: 5pts, basic: 3pts, buggy: 0pts."},
            {"id": 4, "title": "Is visual feedback sufficient?", "maxScore": 5,
             "description": "Check match/complete feedback. Sufficient: 5pts, basic: 3pts, lacking: 0pts."},
            {"id": 5, "title": "Is sound effect design appropriate?", "maxScore": 5,
             "description": "Check audio feedback. Appropriate: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 6, "title": "Is the frame rate stable?", "maxScore": 5,
             "description": "Check performance. Stable: 5pts, acceptable: 3pts, laggy: 0pts."},
            {"id": 7, "title": "Is the operation guidance clear?", "maxScore": 5,
             "description": "Check instructions. Clear: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 8, "title": "How is the replay value?", "maxScore": 5,
             "description": "Check replayability. High: 5pts, medium: 3pts, low: 0pts."},
            {"id": 9, "title": "Is the adaptability good?", "maxScore": 5,
             "description": "Check responsive design. Good: 5pts, basic: 3pts, broken: 0pts."},
        ]
    },
    "game_strategy": {
        "name": "Strategy & Card Games",
        "weight": 0.04,
        "tasks": [
            "Create a Chinese chess (Xiangqi) game with AI opponent",
            "Build a Gomoku (Five in a Row) game with AI",
            "Implement a poker game with Texas Hold'em rules",
            "Create a Dou Dizhu (Fighting the Landlord) card game",
            "Build a Blackjack game with betting system",
            "Develop a Mahjong game with scoring",
            "Build a Reversi/Othello game with AI",
            "Implement a Minesweeper with custom difficulty",
            "Create a Solitaire card game",
            "Build a checkers game with AI",
        ],
        "checklist": [
            {"id": 0, "title": "Is the strategic depth sufficient?", "maxScore": 5,
             "description": "Check game strategy complexity. Sufficient: 5pts, basic: 3pts, shallow: 0pts."},
            {"id": 1, "title": "Is the AI difficulty appropriate?", "maxScore": 5,
             "description": "Check AI challenge level. Appropriate: 5pts, basic: 3pts, poor: 0pts."},
            {"id": 2, "title": "Is collision detection accurate?", "maxScore": 5,
             "description": "Check piece/card placement. Accurate: 5pts, basic: 3pts, broken: 0pts."},
            {"id": 3, "title": "Is game state management correct?", "maxScore": 5,
             "description": "Check game rules and scoring. Correct: 5pts, basic: 3pts, buggy: 0pts."},
            {"id": 4, "title": "Is visual feedback sufficient?", "maxScore": 5,
             "description": "Check move/win feedback. Sufficient: 5pts, basic: 3pts, lacking: 0pts."},
            {"id": 5, "title": "Is sound effect design appropriate?", "maxScore": 5,
             "description": "Check audio feedback. Appropriate: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 6, "title": "Is the frame rate stable?", "maxScore": 5,
             "description": "Check performance. Stable: 5pts, acceptable: 3pts, laggy: 0pts."},
            {"id": 7, "title": "Is the operation guidance clear?", "maxScore": 5,
             "description": "Check instructions. Clear: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 8, "title": "How is the replay value?", "maxScore": 5,
             "description": "Check replayability. High: 5pts, medium: 3pts, low: 0pts."},
            {"id": 9, "title": "Is the adaptability good?", "maxScore": 5,
             "description": "Check responsive design. Good: 5pts, basic: 3pts, broken: 0pts."},
        ]
    },
    "crud_management": {
        "name": "CRUD & Management Systems",
        "weight": 0.12,
        "tasks": [
            "Build a task management system with categories and priorities",
            "Create an employee management system with CRUD operations",
            "Implement a library book management system",
            "Build a student grade management system",
            "Create an inventory management system with stock tracking",
            "Develop a customer relationship management (CRM) system",
            "Build a hospital patient record system",
            "Create a restaurant order management system",
            "Implement a hotel room booking system",
            "Build a project management dashboard with Kanban board",
        ],
        "checklist": [
            {"id": 0, "title": "Is CRUD functionality complete?", "maxScore": 5,
             "description": "Check create/read/update/delete. Complete: 5pts, partial: 3pts, missing: 0pts."},
            {"id": 1, "title": "Is list display reasonable?", "maxScore": 5,
             "description": "Check table/list layout. Reasonable: 5pts, basic: 3pts, poor: 0pts."},
            {"id": 2, "title": "Is form design user-friendly?", "maxScore": 5,
             "description": "Check form UX. User-friendly: 5pts, basic: 3pts, confusing: 0pts."},
            {"id": 3, "title": "Is search and filtering complete?", "maxScore": 5,
             "description": "Check search/filter functions. Complete: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 4, "title": "Is batch operation supported?", "maxScore": 5,
             "description": "Check bulk actions. Supported: 5pts, partial: 3pts, none: 0pts."},
            {"id": 5, "title": "Is pagination functionality correct?", "maxScore": 5,
             "description": "Check pagination. Correct: 5pts, basic: 3pts, broken: 0pts."},
            {"id": 6, "title": "Is import/export supported?", "maxScore": 5,
             "description": "Check data import/export. Supported: 5pts, partial: 3pts, none: 0pts."},
            {"id": 7, "title": "Is operation confirmation complete?", "maxScore": 5,
             "description": "Check delete/edit confirmations. Complete: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 8, "title": "Is status feedback timely?", "maxScore": 5,
             "description": "Check loading/success/error feedback. Timely: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 9, "title": "Is data validation sufficient?", "maxScore": 5,
             "description": "Check input validation. Sufficient: 5pts, basic: 3pts, missing: 0pts."},
        ]
    },
    "crud_business": {
        "name": "Business Applications",
        "weight": 0.10,
        "tasks": [
            "Create a login/registration system with form validation",
            "Build a multi-step form wizard with validation",
            "Implement a file upload system with progress tracking",
            "Create a dynamic form builder with drag-and-drop",
            "Build a survey/questionnaire system",
            "Develop a document approval workflow system",
            "Create a time tracking and attendance system",
            "Build an expense report submission system",
            "Implement a scheduling and appointment system",
            "Create a feedback and rating collection system",
        ],
        "checklist": [
            {"id": 0, "title": "Is the core functionality complete?", "maxScore": 5,
             "description": "Check main features. Complete: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 1, "title": "Is the business process correct?", "maxScore": 5,
             "description": "Check workflow logic. Correct: 5pts, basic: 3pts, wrong: 0pts."},
            {"id": 2, "title": "Is data validation complete?", "maxScore": 5,
             "description": "Check input validation. Complete: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 3, "title": "Is state management reasonable?", "maxScore": 5,
             "description": "Check state handling. Reasonable: 5pts, basic: 3pts, poor: 0pts."},
            {"id": 4, "title": "Is loading state handling complete?", "maxScore": 5,
             "description": "Check loading indicators. Complete: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 5, "title": "Is the navigation structure clear?", "maxScore": 5,
             "description": "Check navigation. Clear: 5pts, basic: 3pts, confusing: 0pts."},
            {"id": 6, "title": "Is responsive adaptation good?", "maxScore": 5,
             "description": "Check responsive design. Good: 5pts, basic: 3pts, broken: 0pts."},
            {"id": 7, "title": "Is operation feedback timely?", "maxScore": 5,
             "description": "Check user feedback. Timely: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 8, "title": "Is data persistence reliable?", "maxScore": 5,
             "description": "Check data saving. Reliable: 5pts, basic: 3pts, buggy: 0pts."},
            {"id": 9, "title": "Are security considerations sufficient?", "maxScore": 5,
             "description": "Check security. Sufficient: 5pts, basic: 3pts, missing: 0pts."},
        ]
    },
    "chart_basic": {
        "name": "Data Charts",
        "weight": 0.06,
        "tasks": [
            "Create an interactive bar chart with tooltips and animations",
            "Build a real-time line chart with streaming data",
            "Implement a pie/donut chart with drill-down capability",
            "Create a stacked area chart for time series data",
            "Build a scatter plot with trend line and zoom",
            "Develop a radar/spider chart for comparison",
            "Create a candlestick chart for stock data",
            "Build a heatmap visualization with color scales",
            "Implement a treemap for hierarchical data",
            "Create a Gantt chart for project timeline",
        ],
        "checklist": [
            {"id": 0, "title": "Is the chart type selection appropriate?", "maxScore": 5,
             "description": "Check chart type for data. Appropriate: 5pts, acceptable: 3pts, wrong: 0pts."},
            {"id": 1, "title": "Is data display accurate?", "maxScore": 5,
             "description": "Check data mapping accuracy. Accurate: 5pts, basic: 3pts, errors: 0pts."},
            {"id": 2, "title": "Are legends and labels clear?", "maxScore": 5,
             "description": "Check legend/axis labels. Clear: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 3, "title": "Is the color scheme professional?", "maxScore": 5,
             "description": "Check color design. Professional: 5pts, acceptable: 3pts, poor: 0pts."},
            {"id": 4, "title": "Are interactive features complete?", "maxScore": 5,
             "description": "Check tooltips/zoom/drill-down. Complete: 5pts, basic: 3pts, none: 0pts."},
            {"id": 5, "title": "Are animation effects appropriate?", "maxScore": 5,
             "description": "Check animations. Appropriate: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 6, "title": "Is responsive adaptation good?", "maxScore": 5,
             "description": "Check responsive design. Good: 5pts, basic: 3pts, broken: 0pts."},
            {"id": 7, "title": "Is data update smooth?", "maxScore": 5,
             "description": "Check data refresh. Smooth: 5pts, basic: 3pts, laggy: 0pts."},
            {"id": 8, "title": "How is performance with large datasets?", "maxScore": 5,
             "description": "Check large data handling. Good: 5pts, acceptable: 3pts, slow: 0pts."},
            {"id": 9, "title": "Is export functionality supported?", "maxScore": 5,
             "description": "Check export feature. Supported: 5pts, partial: 3pts, none: 0pts."},
        ]
    },
    "dashboard": {
        "name": "Dashboards & Analytics",
        "weight": 0.03,
        "tasks": [
            "Create a real-time analytics dashboard with multiple KPIs",
            "Build a sales performance dashboard with charts",
            "Implement a system monitoring dashboard with metrics",
            "Create a social media analytics dashboard",
            "Build a financial portfolio dashboard",
            "Develop an IoT device monitoring dashboard",
            "Create a website traffic analytics dashboard",
            "Build a healthcare metrics dashboard",
            "Implement an energy consumption dashboard",
            "Create a supply chain analytics dashboard",
        ],
        "checklist": [
            {"id": 0, "title": "Is the dashboard layout reasonable?", "maxScore": 5,
             "description": "Check layout design. Reasonable: 5pts, acceptable: 3pts, poor: 0pts."},
            {"id": 1, "title": "Are key metrics highlighted?", "maxScore": 5,
             "description": "Check KPI visibility. Highlighted: 5pts, basic: 3pts, hidden: 0pts."},
            {"id": 2, "title": "Are legends and labels clear?", "maxScore": 5,
             "description": "Check labels/legends. Clear: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 3, "title": "Is the color scheme professional?", "maxScore": 5,
             "description": "Check color design. Professional: 5pts, acceptable: 3pts, poor: 0pts."},
            {"id": 4, "title": "Are interactive features complete?", "maxScore": 5,
             "description": "Check filtering/drill-down. Complete: 5pts, basic: 3pts, none: 0pts."},
            {"id": 5, "title": "Are animation effects appropriate?", "maxScore": 5,
             "description": "Check animations. Appropriate: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 6, "title": "Is responsive adaptation good?", "maxScore": 5,
             "description": "Check responsive design. Good: 5pts, basic: 3pts, broken: 0pts."},
            {"id": 7, "title": "Is data update smooth?", "maxScore": 5,
             "description": "Check data refresh. Smooth: 5pts, basic: 3pts, laggy: 0pts."},
            {"id": 8, "title": "How is performance with large datasets?", "maxScore": 5,
             "description": "Check large data handling. Good: 5pts, acceptable: 3pts, slow: 0pts."},
            {"id": 9, "title": "Is export functionality supported?", "maxScore": 5,
             "description": "Check export feature. Supported: 5pts, partial: 3pts, none: 0pts."},
        ]
    },
    "map_geo": {
        "name": "Maps & Geospatial",
        "weight": 0.02,
        "tasks": [
            "Create an interactive China map with provincial data",
            "Build a world map with country statistics",
            "Implement a heat map overlay on geographic map",
            "Create a route planning map with markers",
            "Build a real-time location tracking map",
            "Develop a choropleth map for population data",
            "Create a map with clustered markers",
            "Build a 3D terrain visualization map",
            "Implement a store locator map with search",
            "Create a delivery tracking map with routes",
        ],
        "checklist": [
            {"id": 0, "title": "Is map rendering correct?", "maxScore": 5,
             "description": "Check map display. Correct: 5pts, basic: 3pts, broken: 0pts."},
            {"id": 1, "title": "Is data mapping accurate?", "maxScore": 5,
             "description": "Check geo data accuracy. Accurate: 5pts, basic: 3pts, wrong: 0pts."},
            {"id": 2, "title": "Are legends and labels clear?", "maxScore": 5,
             "description": "Check map labels. Clear: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 3, "title": "Is the color scheme professional?", "maxScore": 5,
             "description": "Check color design. Professional: 5pts, acceptable: 3pts, poor: 0pts."},
            {"id": 4, "title": "Are interactive features complete?", "maxScore": 5,
             "description": "Check zoom/pan/click. Complete: 5pts, basic: 3pts, none: 0pts."},
            {"id": 5, "title": "Are animation effects appropriate?", "maxScore": 5,
             "description": "Check animations. Appropriate: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 6, "title": "Is responsive adaptation good?", "maxScore": 5,
             "description": "Check responsive design. Good: 5pts, basic: 3pts, broken: 0pts."},
            {"id": 7, "title": "Is data update smooth?", "maxScore": 5,
             "description": "Check data refresh. Smooth: 5pts, basic: 3pts, laggy: 0pts."},
            {"id": 8, "title": "How is performance with large datasets?", "maxScore": 5,
             "description": "Check large data handling. Good: 5pts, acceptable: 3pts, slow: 0pts."},
            {"id": 9, "title": "Is export functionality supported?", "maxScore": 5,
             "description": "Check export feature. Supported: 5pts, partial: 3pts, none: 0pts."},
        ]
    },
    "svg_graphics": {
        "name": "SVG & Graphics",
        "weight": 0.08,
        "tasks": [
            "Create an animated SVG logo with morphing effects",
            "Build an SVG icon set with hover animations",
            "Implement an SVG infographic with animated data",
            "Create an SVG illustration with interactive elements",
            "Build an SVG loading spinner collection",
            "Develop an animated SVG weather icon set",
            "Create an SVG progress indicator with animations",
            "Build an SVG pattern generator",
            "Implement an SVG flowchart with connections",
            "Create an animated SVG background pattern",
        ],
        "checklist": [
            {"id": 0, "title": "Is the SVG structure standardized?", "maxScore": 5,
             "description": "Check SVG markup. Standardized: 5pts, acceptable: 3pts, messy: 0pts."},
            {"id": 1, "title": "Is graphic drawing accurate?", "maxScore": 5,
             "description": "Check shapes/paths. Accurate: 5pts, basic: 3pts, wrong: 0pts."},
            {"id": 2, "title": "Is the style design aesthetically pleasing?", "maxScore": 5,
             "description": "Check visual design. Pleasing: 5pts, acceptable: 3pts, ugly: 0pts."},
            {"id": 3, "title": "Are animation effects smooth?", "maxScore": 5,
             "description": "Check animations. Smooth: 5pts, acceptable: 3pts, choppy: 0pts."},
            {"id": 4, "title": "Is scaling performance good?", "maxScore": 5,
             "description": "Check responsiveness. Good: 5pts, acceptable: 3pts, broken: 0pts."},
            {"id": 5, "title": "Is the code optimized?", "maxScore": 5,
             "description": "Check code efficiency. Optimized: 5pts, basic: 3pts, bloated: 0pts."},
            {"id": 6, "title": "Is the reuse mechanism applied?", "maxScore": 5,
             "description": "Check symbol/defs usage. Applied: 5pts, partial: 3pts, none: 0pts."},
            {"id": 7, "title": "Are interactive effects implemented?", "maxScore": 5,
             "description": "Check hover/click effects. Implemented: 5pts, basic: 3pts, none: 0pts."},
            {"id": 8, "title": "Is semantics considered?", "maxScore": 5,
             "description": "Check accessibility. Considered: 5pts, basic: 3pts, ignored: 0pts."},
            {"id": 9, "title": "How is browser compatibility?", "maxScore": 5,
             "description": "Check cross-browser. Good: 5pts, basic: 3pts, broken: 0pts."},
        ]
    },
    "tool_calculator": {
        "name": "Calculators & Converters",
        "weight": 0.06,
        "tasks": [
            "Create a scientific calculator with advanced functions",
            "Build a mortgage/loan payment calculator",
            "Implement a unit converter (length, weight, temperature)",
            "Create a currency exchange rate calculator",
            "Build a BMI/health metrics calculator",
            "Develop a tip and bill splitting calculator",
            "Create a date/time difference calculator",
            "Build a percentage and ratio calculator",
            "Implement a geometric shape calculator",
            "Create a grade point average (GPA) calculator",
        ],
        "checklist": [
            {"id": 0, "title": "Is the core functionality accurate?", "maxScore": 5,
             "description": "Check calculation accuracy. Accurate: 5pts, mostly: 3pts, wrong: 0pts."},
            {"id": 1, "title": "Is input validation complete?", "maxScore": 5,
             "description": "Check input handling. Complete: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 2, "title": "Is output display clear?", "maxScore": 5,
             "description": "Check result presentation. Clear: 5pts, basic: 3pts, confusing: 0pts."},
            {"id": 3, "title": "Is operation convenient?", "maxScore": 5,
             "description": "Check UX design. Convenient: 5pts, basic: 3pts, awkward: 0pts."},
            {"id": 4, "title": "Are keyboard shortcuts supported?", "maxScore": 5,
             "description": "Check keyboard input. Supported: 5pts, partial: 3pts, none: 0pts."},
            {"id": 5, "title": "Is history saved?", "maxScore": 5,
             "description": "Check calculation history. Saved: 5pts, partial: 3pts, none: 0pts."},
            {"id": 6, "title": "Is copy/export convenient?", "maxScore": 5,
             "description": "Check copy functionality. Convenient: 5pts, basic: 3pts, none: 0pts."},
            {"id": 7, "title": "Is batch processing supported?", "maxScore": 5,
             "description": "Check batch operations. Supported: 5pts, partial: 3pts, none: 0pts."},
            {"id": 8, "title": "Is real-time preview implemented?", "maxScore": 5,
             "description": "Check live preview. Implemented: 5pts, basic: 3pts, none: 0pts."},
            {"id": 9, "title": "Is offline use supported?", "maxScore": 5,
             "description": "Check offline capability. Supported: 5pts, partial: 3pts, none: 0pts."},
        ]
    },
    "media_processing": {
        "name": "Media Processing Tools",
        "weight": 0.06,
        "tasks": [
            "Create an image cropping and resizing tool",
            "Build an image filter editor with presets",
            "Implement an image annotation tool with shapes and text",
            "Create a video player with custom controls",
            "Build an audio waveform visualizer",
            "Develop a watermark adding tool for images",
            "Create an image color picker and palette generator",
            "Build a collage maker with drag-and-drop",
            "Implement a screenshot annotation tool",
            "Create an image comparison slider tool",
        ],
        "checklist": [
            {"id": 0, "title": "Is media processing correct?", "maxScore": 5,
             "description": "Check media operations. Correct: 5pts, basic: 3pts, broken: 0pts."},
            {"id": 1, "title": "Are playback controls complete?", "maxScore": 5,
             "description": "Check preview controls. Complete: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 2, "title": "Is the preview function accurate?", "maxScore": 5,
             "description": "Check real-time preview. Accurate: 5pts, basic: 3pts, wrong: 0pts."},
            {"id": 3, "title": "Is the editing interface intuitive?", "maxScore": 5,
             "description": "Check editor UX. Intuitive: 5pts, basic: 3pts, confusing: 0pts."},
            {"id": 4, "title": "Is undo/redo supported?", "maxScore": 5,
             "description": "Check history support. Supported: 5pts, partial: 3pts, none: 0pts."},
            {"id": 5, "title": "Are effects and filters rich?", "maxScore": 5,
             "description": "Check effect variety. Rich: 5pts, basic: 3pts, limited: 0pts."},
            {"id": 6, "title": "Is the export function complete?", "maxScore": 5,
             "description": "Check export options. Complete: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 7, "title": "How is performance?", "maxScore": 5,
             "description": "Check processing speed. Good: 5pts, acceptable: 3pts, slow: 0pts."},
            {"id": 8, "title": "Is the timeline/track intuitive?", "maxScore": 5,
             "description": "Check timeline UX. Intuitive: 5pts, basic: 3pts, confusing: 0pts."},
            {"id": 9, "title": "Are keyboard shortcuts complete?", "maxScore": 5,
             "description": "Check shortcuts. Complete: 5pts, partial: 3pts, none: 0pts."},
        ]
    },
    "simulation": {
        "name": "Physics Simulations",
        "weight": 0.05,
        "tasks": [
            "Create a gravity simulation with planets and orbits",
            "Build a cloth physics simulation with wind",
            "Implement a particle system with collision",
            "Create a fluid dynamics simulation",
            "Build a pendulum physics simulation",
            "Develop a bouncing ball physics demo",
            "Create a spring and mass simulation",
            "Build a projectile motion simulator",
            "Implement a wave interference simulation",
            "Create a solar system orbital simulation",
        ],
        "checklist": [
            {"id": 0, "title": "Is the simulation logic correct?", "maxScore": 5,
             "description": "Check physics accuracy. Correct: 5pts, basic: 3pts, wrong: 0pts."},
            {"id": 1, "title": "Is parameter control flexible?", "maxScore": 5,
             "description": "Check adjustable params. Flexible: 5pts, limited: 3pts, fixed: 0pts."},
            {"id": 2, "title": "Are visualization effects clear?", "maxScore": 5,
             "description": "Check visual clarity. Clear: 5pts, basic: 3pts, confusing: 0pts."},
            {"id": 3, "title": "Is runtime performance smooth?", "maxScore": 5,
             "description": "Check frame rate. Smooth 60fps: 5pts, 30fps: 3pts, laggy: 0pts."},
            {"id": 4, "title": "Is time control complete?", "maxScore": 5,
             "description": "Check play/pause/speed. Complete: 5pts, basic: 3pts, none: 0pts."},
            {"id": 5, "title": "Can initial conditions be set?", "maxScore": 5,
             "description": "Check initial state config. Yes: 5pts, partial: 3pts, no: 0pts."},
            {"id": 6, "title": "Is data output supported?", "maxScore": 5,
             "description": "Check data export. Supported: 5pts, partial: 3pts, none: 0pts."},
            {"id": 7, "title": "Is interactive control intuitive?", "maxScore": 5,
             "description": "Check interaction UX. Intuitive: 5pts, basic: 3pts, confusing: 0pts."},
            {"id": 8, "title": "Is boundary handling stable?", "maxScore": 5,
             "description": "Check edge cases. Stable: 5pts, basic: 3pts, crashes: 0pts."},
            {"id": 9, "title": "Is the principle explanation clear?", "maxScore": 5,
             "description": "Check documentation. Clear: 5pts, basic: 3pts, missing: 0pts."},
        ]
    },
    "animation": {
        "name": "Animation & Visual Effects",
        "weight": 0.04,
        "tasks": [
            "Create a morphing shape animation with smooth transitions",
            "Build a particle explosion effect on click",
            "Implement a loading animation with creative design",
            "Create a page transition animation library",
            "Build a text reveal animation with multiple styles",
            "Develop a scroll-triggered animation sequence",
            "Create a cursor trail effect with particles",
            "Build a glitch text effect animation",
            "Implement a liquid fill animation",
            "Create a 3D card flip animation",
        ],
        "checklist": [
            {"id": 0, "title": "Are animation effects smooth?", "maxScore": 5,
             "description": "Check animation fluidity. Smooth: 5pts, acceptable: 3pts, choppy: 0pts."},
            {"id": 1, "title": "Does the visual style meet requirements?", "maxScore": 5,
             "description": "Check design quality. Meets: 5pts, basic: 3pts, poor: 0pts."},
            {"id": 2, "title": "Is the interaction experience smooth?", "maxScore": 5,
             "description": "Check responsiveness. Smooth: 5pts, acceptable: 3pts, laggy: 0pts."},
            {"id": 3, "title": "Is responsive adaptation complete?", "maxScore": 5,
             "description": "Check screen sizes. Complete: 5pts, partial: 3pts, broken: 0pts."},
            {"id": 4, "title": "Is the technical implementation standardized?", "maxScore": 5,
             "description": "Check code quality. Standardized: 5pts, basic: 3pts, messy: 0pts."},
            {"id": 5, "title": "Is performance excellent?", "maxScore": 5,
             "description": "Check frame rate. Excellent: 5pts, acceptable: 3pts, poor: 0pts."},
            {"id": 6, "title": "Are details handled well?", "maxScore": 5,
             "description": "Check polish level. Well: 5pts, basic: 3pts, rough: 0pts."},
            {"id": 7, "title": "Does the creative implementation have highlights?", "maxScore": 5,
             "description": "Check creativity. Highlights: 5pts, standard: 3pts, none: 0pts."},
            {"id": 8, "title": "Is user guidance clear?", "maxScore": 5,
             "description": "Check instructions. Clear: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 9, "title": "Is code structure clear?", "maxScore": 5,
             "description": "Check code organization. Clear: 5pts, basic: 3pts, messy: 0pts."},
        ]
    },
    "communication": {
        "name": "Messaging & Communication",
        "weight": 0.03,
        "tasks": [
            "Create a real-time chat interface with messages",
            "Build a notification center with types",
            "Implement a comment system with replies",
            "Create a social feed with posts and reactions",
            "Build an email inbox interface",
            "Develop a live chat widget",
            "Create a discussion forum interface",
            "Build a contact form with validation",
            "Implement a messaging app interface",
            "Create a feedback collection widget",
        ],
        "checklist": [
            {"id": 0, "title": "Is the messaging feature complete?", "maxScore": 5,
             "description": "Check message functions. Complete: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 1, "title": "Does real-time performance meet requirements?", "maxScore": 5,
             "description": "Check update speed. Meets: 5pts, acceptable: 3pts, slow: 0pts."},
            {"id": 2, "title": "Is data validation complete?", "maxScore": 5,
             "description": "Check input validation. Complete: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 3, "title": "Is state management reasonable?", "maxScore": 5,
             "description": "Check message state. Reasonable: 5pts, basic: 3pts, buggy: 0pts."},
            {"id": 4, "title": "Is loading state handling complete?", "maxScore": 5,
             "description": "Check loading feedback. Complete: 5pts, basic: 3pts, none: 0pts."},
            {"id": 5, "title": "Is the navigation structure clear?", "maxScore": 5,
             "description": "Check navigation. Clear: 5pts, basic: 3pts, confusing: 0pts."},
            {"id": 6, "title": "Is responsive adaptation good?", "maxScore": 5,
             "description": "Check responsive design. Good: 5pts, basic: 3pts, broken: 0pts."},
            {"id": 7, "title": "Is operation feedback timely?", "maxScore": 5,
             "description": "Check user feedback. Timely: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 8, "title": "Is data persistence reliable?", "maxScore": 5,
             "description": "Check data saving. Reliable: 5pts, basic: 3pts, buggy: 0pts."},
            {"id": 9, "title": "Are security considerations sufficient?", "maxScore": 5,
             "description": "Check security. Sufficient: 5pts, basic: 3pts, missing: 0pts."},
        ]
    },
    "ecommerce": {
        "name": "E-commerce & Shopping",
        "weight": 0.02,
        "tasks": [
            "Create a product listing page with filters",
            "Build a shopping cart with quantity controls",
            "Implement a product detail page with gallery",
            "Create a checkout flow with form validation",
            "Build a product comparison tool",
            "Develop a wishlist management interface",
            "Create a product quick view modal",
            "Build an order tracking interface",
            "Implement a product review and rating system",
            "Create a promotional banner slider",
        ],
        "checklist": [
            {"id": 0, "title": "Is product display complete?", "maxScore": 5,
             "description": "Check product info display. Complete: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 1, "title": "Is the shopping process smooth?", "maxScore": 5,
             "description": "Check cart/checkout flow. Smooth: 5pts, basic: 3pts, awkward: 0pts."},
            {"id": 2, "title": "Is data validation complete?", "maxScore": 5,
             "description": "Check form validation. Complete: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 3, "title": "Is state management reasonable?", "maxScore": 5,
             "description": "Check cart state. Reasonable: 5pts, basic: 3pts, buggy: 0pts."},
            {"id": 4, "title": "Is loading state handling complete?", "maxScore": 5,
             "description": "Check loading feedback. Complete: 5pts, basic: 3pts, none: 0pts."},
            {"id": 5, "title": "Is the navigation structure clear?", "maxScore": 5,
             "description": "Check navigation. Clear: 5pts, basic: 3pts, confusing: 0pts."},
            {"id": 6, "title": "Is responsive adaptation good?", "maxScore": 5,
             "description": "Check responsive design. Good: 5pts, basic: 3pts, broken: 0pts."},
            {"id": 7, "title": "Is operation feedback timely?", "maxScore": 5,
             "description": "Check user feedback. Timely: 5pts, basic: 3pts, missing: 0pts."},
            {"id": 8, "title": "Is data persistence reliable?", "maxScore": 5,
             "description": "Check data saving. Reliable: 5pts, basic: 3pts, buggy: 0pts."},
            {"id": 9, "title": "Are security considerations sufficient?", "maxScore": 5,
             "description": "Check security. Sufficient: 5pts, basic: 3pts, missing: 0pts."},
        ]
    },
}


# =============================================================================
# Tech Stacks
# =============================================================================

TECH_STACKS = {
    "canvas": {
        "name": "HTML5 Canvas + JavaScript",
        "desc": "Canvas-based rendering for graphics and animations",
        "role": "simulation development expert proficient in physics engines and mathematical modeling",
        "skills": "Physics simulation (gravity, collision, fluid), Particle systems, Numerical computation and visualization, WebGL shaders, Performance optimization",
    },
    "vanilla": {
        "name": "HTML + CSS + JavaScript",
        "desc": "Pure vanilla implementation without frameworks",
        "role": "experienced frontend developer",
        "skills": "HTML5/CSS3, JavaScript ES6+, DOM manipulation, CSS animations, Responsive design",
    },
    "svg": {
        "name": "SVG + CSS Animation",
        "desc": "Vector graphics with CSS animations",
        "role": "SVG and vector graphics specialist",
        "skills": "SVG path manipulation, CSS keyframe animations, Transform animations, Gradient and filter effects",
    },
    "threejs": {
        "name": "Three.js",
        "desc": "3D graphics with Three.js library",
        "role": "3D graphics and WebGL expert",
        "skills": "Three.js scene setup, 3D modeling, Lighting and materials, Camera controls, Performance optimization",
    },
    "react": {
        "name": "React + TypeScript",
        "desc": "Component-based architecture with TypeScript",
        "role": "React and TypeScript expert",
        "skills": "React hooks and state management, TypeScript interfaces, Component architecture, Performance optimization",
    },
    "echarts": {
        "name": "ECharts",
        "desc": "Data visualization with ECharts library",
        "role": "data visualization expert",
        "skills": "ECharts configuration, Chart customization, Interactive features, Animation effects",
    },
}


# =============================================================================
# Prompt Generation
# =============================================================================

def generate_question(task: str, tech_stack: Dict) -> str:
    """Generate a detailed question with role and requirements."""
    question = f"""======== ROLE SETTING ========
You are a {tech_stack['role']}. Your skills include:
- {tech_stack['skills'].replace(', ', chr(10) + '- ')}

======== TASK BACKGROUND ========
You need to use {tech_stack['name']} technology stack to complete the following front-end development task.
Please implement using {tech_stack['desc']} and integrate all code into a single HTML file.

======== PROJECT REQUIREMENTS ========
Make sure the code you generate is executable for demonstration purposes. {task}

======== TECHNICAL REQUIREMENTS ========
1. Use {tech_stack['name']} to implement all functionality
2. The code must be complete and runnable, with no parts omitted
3. Add clear comments explaining key logic
4. Implement responsive design to adapt to different screen sizes
5. Include necessary error handling and user feedback

======== OUTPUT FORMAT ========
Please first briefly explain your implementation approach in 2-3 sentences, then output the complete code. The code format is as follows:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Project</title>
    <style>
        /* CSS styles */
    </style>
</head>
<body>
    <!-- HTML content -->
    <script>
        // JavaScript logic
    </script>
</body>
</html>
```

Please ensure the code can be directly copied and run to demonstrate the complete functionality."""

    return question


def build_full_checklist(category_checklist: List[Dict]) -> List[Dict]:
    """Build full 20-item checklist: category items (0-9) + universal items (10-19)."""
    full_checklist = category_checklist.copy()  # ID 0-9 (10 items)
    full_checklist.extend(UNIVERSAL_CHECKLIST)   # ID 10-19 (10 items)
    return full_checklist


def generate_variants(target_count: int) -> List[Dict]:
    """Generate target_count variants based on category weights."""
    variants = []
    variant_id = 0

    # Calculate samples per category based on weights
    category_samples = {}
    for cat_key, cat_info in TASK_CATEGORIES.items():
        count = int(target_count * cat_info['weight'])
        category_samples[cat_key] = count

    # Adjust to reach exact target
    total = sum(category_samples.values())
    diff = target_count - total
    if diff > 0:
        for cat_key in list(TASK_CATEGORIES.keys())[:diff]:
            category_samples[cat_key] += 1

    print(f"Category distribution:")
    for cat_key, count in category_samples.items():
        print(f"  {cat_key}: {count}")
    print()

    # Generate variants for each category
    tech_keys = list(TECH_STACKS.keys())

    for cat_key, sample_count in category_samples.items():
        cat_info = TASK_CATEGORIES[cat_key]
        tasks = cat_info['tasks']
        checklist = cat_info['checklist']

        for i in range(sample_count):
            task = tasks[i % len(tasks)]
            tech_key = tech_keys[i % len(tech_keys)]
            tech_stack = TECH_STACKS[tech_key]

            question = generate_question(task, tech_stack)
            full_checklist = build_full_checklist(checklist)

            record = {
                "index": variant_id,
                "question": question,
                "checklist": full_checklist,
                "category": cat_key,
                "category_name": cat_info['name'],
                "tech_stack": tech_key,
                "base_task": task,
            }

            variants.append(record)
            variant_id += 1

            if variant_id % 200 == 0:
                print(f"  Generated {variant_id}/{target_count} variants...")

    random.shuffle(variants)
    for i, v in enumerate(variants):
        v['index'] = i

    return variants


def main():
    parser = argparse.ArgumentParser(description="Generate 2000 RL prompts")
    parser.add_argument("--output", type=str, default="rl_prompts_2000.jsonl")
    parser.add_argument("--target_count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("Generate RL Prompts Based on Real Categories")
    print("=" * 60)
    print(f"Output: {args.output}")
    print(f"Target count: {args.target_count}")
    print("=" * 60)

    print(f"\nGenerating {args.target_count} prompt variants...")
    variants = generate_variants(args.target_count)

    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for record in variants:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"  Generated prompts: {len(variants)}")
    print(f"  Output: {args.output}")

    cat_counts = {}
    for v in variants:
        cat = v['category_name']
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print(f"\n  Category breakdown:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    if variants:
        sample = variants[0]
        print(f"\n  Sample record:")
        print(f"    Question length: {len(sample['question'])} chars")
        print(f"    Checklist items: {len(sample['checklist'])}")
        print(f"    Max score: {sum(c['maxScore'] for c in sample['checklist'])}")

    print("=" * 60)


if __name__ == "__main__":
    main()
