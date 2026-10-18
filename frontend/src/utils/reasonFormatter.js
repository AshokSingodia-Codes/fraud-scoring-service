/**
 * Plain-English explanations for ML feature names and contributions.
 * Translates complex feature engineer tags into statements any user can understand.
 */

const FEATURE_EXPLANATIONS = {
  "Transaction Amount": {
    increases: "The transaction amount is unusually high compared to typical purchases for this card profile.",
    lowers: "The transaction amount is consistent with normal, low-risk purchase sizes."
  },
  "Log Transaction Amount": {
    increases: "The order total falls into a higher-risk spend bracket.",
    lowers: "The order size is well within customary spending patterns."
  },
  "Card1 Frequency": {
    increases: "This specific card number appears rarely or for the first time in our system, indicating a new or unfamiliar account.",
    lowers: "This card number has a strong history of regular, safe transactions."
  },
  "Card2 Frequency": {
    increases: "The bank issuing this card has a relatively low transaction volume in recent records.",
    lowers: "The bank issuing this card has a proven track record of legitimate activity."
  },
  "Card & Address Frequency": {
    increases: "This specific combination of card number and billing ZIP/address code is seldom seen together.",
    lowers: "This card and billing address pair has been verified across multiple previous legitimate orders."
  },
  "Address1 Frequency": {
    increases: "The billing location or postal region is rarely encountered in safe order histories.",
    lowers: "The billing location corresponds to a widely recognized and active postal hub."
  },
  "Purchaser Email Provider": {
    increases: "The email domain used by the purchaser has a higher historical association with chargeback disputes.",
    lowers: "The purchaser's email provider is a reputable, established provider with low fraud rates."
  },
  "Recipient Email Provider": {
    increases: "The delivery recipient email domain is unusual or associated with temporary email patterns.",
    lowers: "The recipient email provider is standard and matches verified consumer accounts."
  },
  "Email Domain Match Flag": {
    increases: "The purchaser's email domain does not match the delivery recipient's email domain.",
    lowers: "The purchaser and recipient email domains match, indicating the buyer and receiver are identical."
  },
  "Round Amount Flag": {
    increases: "The transaction is an exact whole-dollar amount (e.g. $100.00), which fraudsters often use when testing stolen credentials.",
    lowers: "The transaction contains typical non-round cents (e.g. $24.99), typical of authentic merchant checkouts."
  },
  "Amount Cents Part": {
    increases: "The decimal cent pattern matches card-testing automation scripts.",
    lowers: "The decimal cent value reflects standard retail tax and pricing calculation."
  },
  "Hour of Day": {
    increases: "The purchase occurred during off-peak night hours when unauthorized card usage is statistically elevated.",
    lowers: "The purchase occurred during normal daytime commercial hours."
  },
  "Day of Week": {
    increases: "The timing aligns with weekends or holiday windows when fraud attempts spike.",
    lowers: "The timing aligns with ordinary weekday retail traffic."
  },
  "Identity Provided Flag": {
    increases: "No digital device or browser identity fingerprint was supplied during checkout.",
    lowers: "Verified device details and browser fingerprints were provided, confirming user authenticity."
  },
  "Missing Identity Fields Count": {
    increases: "Multiple standard checkout fields (like browser version or device info) were left blank or omitted.",
    lowers: "All standard technical checkout parameters were thoroughly populated."
  }
};

/**
 * Format a raw reason code item into plain English for normal users.
 */
export function formatReasonExplanation(reason) {
  const label = reason.label || reason.feature;
  const isIncrease = reason.direction === "increases_risk" || reason.contribution > 0;
  
  const template = FEATURE_EXPLANATIONS[label];
  if (template) {
    return isIncrease ? template.increases : template.lowers;
  }

  // Fallback for raw or IEEE-CIS encoded columns (e.g. C1, V258, D2)
  if (isIncrease) {
    return `${label} is higher than normal for this account type, which pushed the risk score up.`;
  } else {
    return `${label} is within safe parameters, helping keep the risk score down.`;
  }
}

/**
 * Get human-readable description for decision actions.
 */
export function getDecisionBadgeInfo(decision) {
  switch (decision?.toLowerCase()) {
    case 'approve':
      return {
        label: 'Approved (Safe)',
        description: 'Low fraud probability. Safe to complete immediately with no added friction.',
        colorClass: 'badge-approve',
        bgLight: 'bg-emerald-50',
        border: 'border-emerald-200',
        text: 'text-emerald-800',
        iconColor: 'text-emerald-600',
        actionAdvice: 'Transaction cleared. No human review needed.'
      };
    case 'review':
      return {
        label: 'Manual Review Needed',
        description: 'Moderate fraud signals detected. An operations analyst should inspect details before releasing goods.',
        colorClass: 'badge-review',
        bgLight: 'bg-amber-50',
        border: 'border-amber-200',
        text: 'text-amber-800',
        iconColor: 'text-amber-600',
        actionAdvice: 'Hold funds temporarily. Verify cardholder identity or phone confirmation.'
      };
    case 'block':
    default:
      return {
        label: 'Blocked (High Risk)',
        description: 'Strong probability of fraudulent activity. Decline to prevent financial chargeback loss.',
        colorClass: 'badge-block',
        bgLight: 'bg-rose-50',
        border: 'border-rose-200',
        text: 'text-rose-800',
        iconColor: 'text-rose-600',
        actionAdvice: 'Decline order immediately and alert security team.'
      };
  }
}
