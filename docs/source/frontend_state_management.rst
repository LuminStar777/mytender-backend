Frontend State Management
=======================

This page documents the state management approaches used in the Spark AI Chatbot frontend application.

Overview
--------

The Spark AI Chatbot frontend uses multiple state management strategies depending on the scope and complexity of the data being managed. The application follows modern React patterns, with a combination of local component state, React Context API, and browser storage mechanisms.

State Management Flow
--------------------

The following diagram illustrates how state flows through the application:

.. figure:: _static/react_state_management.png
   :alt: React State Management Flow
   :align: center
   :width: 100%

   React State Management Flow Diagram

State Management Hierarchy
-------------------------

.. code-block:: text

    ├── Global State
    │   ├── Authentication State (React Auth Kit)
    │   ├── Bid Context (BidContext)
    │   └── UI State (StatusLabelsProvider)
    ├── Feature-Level State
    │   ├── Component Context (TabProvider, etc.)
    │   └── URL Parameters (React Router)
    └── Component-Level State
        ├── React useState Hooks
        ├── React useReducer (for complex state logic)
        └── Local Component State

Context API Implementation
-------------------------

The application uses React Context API extensively for sharing state between components without prop drilling. Key contexts include:

BidContext
^^^^^^^^^^

The ``BidContext`` is the primary state management system for bid-related features. It manages the state of the current bid, including sections, contributions, and editing status.

Implementation:

.. code-block:: typescript

    // BidContext definition
    export const BidContext = createContext({
      sharedState: initialState,
      setSharedState: () => {},
      getBackgroundInfo: () => ({}),
      // ... other methods
    });

    // BidContext provider in BidWritingStateManagerView
    const [sharedState, setSharedState] = useState(initialState);
    
    /* ... state management logic ... */
    
    const contextValue = {
      sharedState,
      setSharedState,
      saveProposal,
      getBackgroundInfo,
      // ... other methods
    };
    
    return (
      <BidContext.Provider value={contextValue}>
        {children}
      </BidContext.Provider>
    );

Usage across components:

.. code-block:: typescript

    // In QuestionCrafter.tsx
    const { sharedState, setSharedState, getBackgroundInfo } = 
      useContext(BidContext);
    const { contributors, outline } = sharedState;
    
    // Updating shared state
    const updatedOutline = outline.map((s) =>
      s.section_id === section.section_id ? { ...s, question: newText } : s
    );
    
    setSharedState((prev) => ({
      ...prev,
      outline: updatedOutline
    }));

StatusLabelsProvider
^^^^^^^^^^^^^^^^^^^

Manages bid status labels and colors throughout the application:

.. code-block:: typescript

    export const StatusLabelsContext = createContext({
      statusLabels: {},
      statusOptions: [],
      // ... other properties
    });
    
    export const StatusLabelsProvider = ({ children }) => {
      const [statusLabels, setStatusLabels] = useState({...});
      
      /* ... state management logic ... */
      
      return (
        <StatusLabelsContext.Provider value={{ statusLabels, statusOptions }}>
          {children}
        </StatusLabelsContext.Provider>
      );
    };

TabProvider
^^^^^^^^^^

Manages tab state for multi-tab interfaces:

.. code-block:: typescript

    export const TabContext = createContext({
      activeTab: "tab1",
      setActiveTab: () => {},
    });
    
    export const TabProvider = ({ children }) => {
      const [activeTab, setActiveTab] = useState("tab1");
      
      return (
        <TabContext.Provider value={{ activeTab, setActiveTab }}>
          {children}
        </TabContext.Provider>
      );
    };

Component-Level State
--------------------

Component-level state is managed using React's built-in hooks, primarily ``useState`` and ``useEffect``.

Q&A Generator Example
^^^^^^^^^^^^^^^^^^^^

The Q&A Generator component demonstrates complex state management with multiple states for different UI elements and features:

.. code-block:: typescript

    // Q&AGenerator.tsx
    const [inputText, setInputText] = useState(
      localStorage.getItem("inputText") || ""
    );
    
    const [responseEditorState, setResponseEditorState] = useState(
      EditorState.createWithContent(
        ContentState.createFromText(localStorage.getItem("response") || "")
      )
    );
    
    const [selectedBidId, setSelectedBidId] = useState("");
    const [selectedFolders, setSelectedFolders] = useState(["default"]);
    
    // Copilot-related state
    const [isCopilotVisible, setIsCopilotVisible] = useState(false);
    const [selectedText, setSelectedText] = useState("");
    const [copilotOptions, setCopilotOptions] = useState([]);
    
    // Message history
    const [messages, setMessages] = useState(() => {
      const savedMessages = localStorage.getItem("messages");
      if (savedMessages) {
        const parsedMessages = JSON.parse(savedMessages);
        if (parsedMessages.length > 0) {
          return parsedMessages;
        }
      }
      return [{ type: "bot", text: "Welcome to Bid Pilot!" }];
    });

Form State Management
^^^^^^^^^^^^^^^^^^^^^

Form state is typically managed with ``useState`` hooks, with useEffect for validation and side effects:

.. code-block:: typescript

    const [inputValue, setInputValue] = useState("");
    const [isValid, setIsValid] = useState(true);
    
    useEffect(() => {
      // Validation logic
      setIsValid(inputValue.length > 0);
    }, [inputValue]);

Persistence Strategies
---------------------

The application uses several strategies to persist state:

Browser LocalStorage
^^^^^^^^^^^^^^^^^^^

Used for persisting user preferences, draft content, and session data:

.. code-block:: typescript

    // Persisting editor content
    useEffect(() => {
      localStorage.setItem(
        "response",
        convertToRaw(responseEditorState.getCurrentContent())
          .blocks.map((block) => block.text)
          .join("\n")
      );
    }, [responseEditorState]);
    
    // Retrieving from localStorage
    const [inputText, setInputText] = useState(
      localStorage.getItem("inputText") || ""
    );

Authentication Tokens
^^^^^^^^^^^^^^^^^^^^

React Auth Kit handles authentication token persistence:

.. code-block:: typescript

    <AuthProvider authType={"localstorage"} authName={"sparkaichatbot"}>
      <AppContent />
    </AuthProvider>

URL Parameters
^^^^^^^^^^^^^

State that needs to be shareable or bookmarkable is stored in URL parameters:

.. code-block:: typescript

    // Reading URL parameters
    const location = useLocation();
    const { bid_id, section: locationSection } = location.state || {};
    
    // Setting URL parameters
    navigate(`/bid/${bid_id}`, { 
      state: { section: updatedSection } 
    });

State Synchronization with Backend
---------------------------------

State is synchronized with the backend through API calls:

.. code-block:: typescript

    const handleSaveProposal = async () => {
      setIsLoading(true);
      try {
        await saveProposal(); // Context method that makes API call
        displayAlert("Proposal saved successfully", "success");
      } catch (error) {
        displayAlert("Failed to save proposal", "danger");
      }
      setIsLoading(false);
    };

Common State Management Patterns
-------------------------------

The application follows several common patterns:

Debounced Updates
^^^^^^^^^^^^^^^^

For preventing excessive API calls during typing:

.. code-block:: typescript

    const typingTimeout = useRef(null);
    
    const handleInputChange = (value) => {
      setInputValue(value);
      
      if (typingTimeout.current) {
        clearTimeout(typingTimeout.current);
      }
      
      typingTimeout.current = setTimeout(() => {
        saveToBackend(value);
      }, 500);
    };

Optimistic Updates
^^^^^^^^^^^^^^^^^

For providing immediate UI feedback before API confirmation:

.. code-block:: typescript

    const updateStatus = async (status) => {
      // Immediately update UI
      setSectionStatus(status);
      
      try {
        // Update shared state
        const updatedOutline = outline.map((s) =>
          s.section_id === section.section_id ? { ...s, status } : s
        );
        
        setSharedState((prev) => ({
          ...prev,
          outline: updatedOutline
        }));
        
        // API call happens in background
      } catch (err) {
        // Revert on error
        setSectionStatus(section.status);
        displayAlert("Failed to update status", "danger");
      }
    };

Analytics Integration
--------------------

State changes often trigger analytics events:

.. code-block:: typescript

    const handleSaveProposal = async () => {
      setIsLoading(true);
      posthog.capture("proposal_save_started", {
        bidId: sharedState.object_id,
        bidName: bidInfo
      });
      
      try {
        await saveProposal();
        posthog.capture("proposal_save_succeeded", {
          bidId: sharedState.object_id,
          bidName: bidInfo
        });
      } catch (error) {
        posthog.capture("proposal_save_failed", {
          bidId: sharedState.object_id,
          bidName: bidInfo,
          error: error.message
        });
      }
      setIsLoading(false);
    };

Best Practices
-------------

The application follows these state management best practices:

1. **Keep state as local as possible** - Use component state for UI-specific data
2. **Lift state up** - When multiple components need the same state
3. **Use context for shared state** - Avoid prop drilling for deeply nested components
4. **Normalize complex state** - Organize nested objects and arrays for easier updates
5. **Memoize expensive calculations** - Use useMemo for derived state that requires computation
6. **Batch related state** - Use useReducer for state with multiple related values
7. **Persist only what's necessary** - Be selective about what gets saved to localStorage/cookies 