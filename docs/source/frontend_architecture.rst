Frontend Architecture
===================

This section documents the frontend architecture of the Spark AI Chatbot, built with React and TypeScript.

Overview
--------

The Spark AI Chatbot frontend is a modern single-page application (SPA) built with React and TypeScript. It provides a rich user interface for interacting with the AI chatbot, managing document libraries, and generating tender responses.

Component Hierarchy
------------------

The React application follows a component-based architecture with clear separation of concerns.

.. image:: _static/react_component_hierarchy.png
   :width: 100%
   :alt: React Component Hierarchy

Tech Stack
----------

The frontend is built using the following core technologies:

* **React**: UI library for building component-based interfaces
* **TypeScript**: Adds static typing to JavaScript for better developer experience
* **React Router**: Handles navigation and routing within the application
* **React Auth Kit**: Manages authentication state and token management
* **PostHog**: Analytics tracking
* **React-Toastify**: Toast notifications for user feedback

Directory Structure
------------------

The frontend application follows a modular structure:

.. code-block:: text

    src/
    ├── App.tsx                  # Main application component
    ├── Widget.css               # Global widget styles
    ├── components/              # Shared UI components
    │   ├── UpdateChecker.tsx    # Component to check for app updates
    │   └── ...
    ├── resources/               # Static resources
    │   └── manrope.css          # Font definitions
    ├── routes/                  # Application routing
    │   └── Routing.tsx          # Main routing configuration
    ├── views/                   # Page components
    │   ├── Bids/                # Bid management views
    │   │   ├── components/      # Bid-specific components
    │   │   │   └── BidStatusMenu.tsx
    │   │   └── ...
    │   ├── Chat/                # Chat interface views
    │   ├── Library/             # Document library views
    │   └── ...
    ├── services/                # API integrations and services
    └── utils/                   # Utility functions and helpers

Application Entry Point
----------------------

The application is bootstrapped in ``App.tsx``, which sets up the core providers:

.. code-block:: typescript

    const App = () => {
      return (
        <AuthProvider authType={"localstorage"} authName={"sparkaichatbot"}>
          <AppContent />
          <ToastContainer
            position="bottom-right"
            autoClose={4000}
            newestOnTop={false}
            closeOnClick={false}
            rtl={false}
            pauseOnFocusLoss
            draggable={false}
            hideProgressBar={true}
          />
        </AuthProvider>
      );
    };

Authentication
-------------

Authentication is handled via React Auth Kit, which manages:

* Token storage (using localStorage)
* Authentication state management
* Session expiration
* Protected routes

All authenticated content is wrapped in the ``AuthProvider`` component:

.. code-block:: typescript

    <AuthProvider authType={"localstorage"} authName={"sparkaichatbot"}>
      <AppContent />
    </AuthProvider>

Routing
-------

The application uses React Router for navigation. Routes are defined in ``Routing.tsx`` and include:

* Public routes (login, signup, forgot password)
* Protected routes requiring authentication
* Role-based routes with specific permission requirements

Context Providers
----------------

The application uses several React Context providers to manage global state:

* ``StatusLabelsProvider``: Manages bid status labels and colors
* ``AuthProvider``: Manages authentication state
* Other context providers for specific feature areas

Analytics Integration
--------------------

The application integrates with Google Analytics (GA4) and PostHog for user behavior tracking:

.. code-block:: typescript

    // Google Analytics
    ReactGA4.initialize("G-X8S1ZMRM3C");

    // PostHog
    posthog.init("phc_bdUxtNoJmZWNnu1Ar29zUtusFQ4bvU91fZpLw5v4Y3e", {
      api_host: "https://eu.i.posthog.com",
      person_profiles: "identified_only"
    });

Key UI Components
----------------

The frontend includes several key UI component types:

1. **Page Components**: Full-page views like dashboard, chat interface, etc.
2. **Feature Components**: Self-contained feature blocks (bid editor, document uploader)
3. **Common UI Components**: Reusable UI elements like buttons, input fields
4. **Layout Components**: Structural elements like headers, sidebars, and navigation

Component Design Patterns
------------------------

The frontend follows several React component design patterns:

1. **Container/Presenter Pattern**: Separates data fetching logic from presentation
2. **Context + Hooks Pattern**: Uses React Context for state management and custom hooks for reusable logic
3. **Compound Components**: Creates intuitive component APIs for complex UI elements

API Communication
---------------

The frontend communicates with the backend API using:

1. **Fetch API**: For general API requests
2. **FormData**: For file uploads and multipart requests
3. **WebSockets** (where applicable): For real-time features

State Management
--------------

State is managed at different levels:

1. **Component State**: Using React's ``useState`` for component-specific state
2. **Context API**: For sharing state between components in a specific feature
3. **URL Parameters**: For state that should be reflected in the URL
4. **LocalStorage**: For persisting certain user preferences and session data

Development Guidelines
--------------------

When contributing to the frontend, developers should follow these guidelines:

1. **TypeScript Usage**: All new components should be written in TypeScript with proper type definitions
2. **Component Structure**: 
   - One component per file
   - Use functional components with hooks
   - Export components as named exports
3. **Styling**: 
   - Use CSS modules for component-specific styles
   - Follow the established design system
4. **State Management**:
   - Minimize prop drilling by using context appropriately
   - Keep state as local as possible
5. **Performance**:
   - Use React.memo for pure components
   - Implement virtualization for long lists
   - Use lazy loading for routes and heavy components 